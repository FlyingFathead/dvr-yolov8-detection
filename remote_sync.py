# remote_sync.py

import os
import subprocess
import logging
import threading
import time
from queue import Queue, Empty

# Check for paramiko
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    paramiko = None
    PARAMIKO_AVAILABLE = False

class RemoteSync:
    def __init__(self, config, main_logger, save_dir_base, aggregated_detections_file):
        """
        Initializes the RemoteSync class.

        :param config: ConfigParser object containing configuration settings.
        :param main_logger: The main logger instance from the main program.
        :param save_dir_base: Base directory where detection images are saved.
        :param aggregated_detections_file: Path to the aggregated detections JSON file.
        """
        self.config = config
        self.logger = logging.getLogger("remote_sync")  # Dedicated logger for remote_sync
        self.save_dir_base = save_dir_base
        self.aggregated_detections_file = aggregated_detections_file

        # Remote Sync Configuration
        self.REMOTE_SYNC_ENABLED = config.getboolean('remote_sync', 'enabled', fallback=True)
        self.USE_PARAMIKO = config.getboolean('remote_sync', 'use_paramiko', fallback=True)
        self.READ_REMOTE_CONFIG_FROM_ENV = config.getboolean('remote_sync', 'read_remote_config_from_env', fallback=True)
        self.REMOTE_USER_ENV_VAR = config.get('remote_sync', 'remote_user_env_var', fallback='DVR_YOLOV8_REMOTE_USER')
        self.REMOTE_HOST_ENV_VAR = config.get('remote_sync', 'remote_host_env_var', fallback='DVR_YOLOV8_REMOTE_HOST')
        self.REMOTE_DIR_ENV_VAR = config.get('remote_sync', 'remote_dir_env_var', fallback='DVR_YOLOV8_REMOTE_DIR')
        self.REMOTE_USER = config.get('remote_sync', 'remote_user', fallback=None)
        self.REMOTE_HOST = config.get('remote_sync', 'remote_host', fallback=None)
        self.REMOTE_DIR = config.get('remote_sync', 'remote_dir', fallback=None)
        self.REMOTE_SSH_KEY = config.get('remote_sync', 'remote_ssh_key', fallback=None)
        self.SYNC_AGGREGATED_DETECTIONS = config.getboolean('remote_sync', 'sync_aggregated_detections', fallback=True)
        self.SYNC_DETECTION_AREA_IMAGES = config.getboolean('remote_sync', 'sync_detection_area_images', fallback=True)
        self.SYNC_FULL_FRAME_IMAGES = config.getboolean('remote_sync', 'sync_full_frame_images', fallback=False)

        # Retry Configuration
        self.MAX_RETRIES = config.getint('remote_sync', 'max_retries', fallback=3)
        self.RETRY_DELAY = config.getint('remote_sync', 'retry_delay', fallback=5)

        # Queue Configuration
        self.REMOTE_SYNC_QUEUE_MAXSIZE = config.getint('remote_sync', 'remote_sync_queue_maxsize', fallback=1000)
        self.remote_sync_queue = Queue(maxsize=self.REMOTE_SYNC_QUEUE_MAXSIZE)
        self.remote_sync_stop_event = threading.Event()
        self.remote_sync_thread = None

        # Handle Remote Sync Configuration
        if self.READ_REMOTE_CONFIG_FROM_ENV:
            # Read from environment variables
            self.REMOTE_USER = os.getenv(self.REMOTE_USER_ENV_VAR)
            self.REMOTE_HOST = os.getenv(self.REMOTE_HOST_ENV_VAR)
            self.REMOTE_DIR = os.getenv(self.REMOTE_DIR_ENV_VAR)
        else:
            # Read from config.ini directly
            # self.REMOTE_USER, self.REMOTE_HOST, self.REMOTE_DIR are already set from config.get above
            pass

        # Detailed logging for what's available and what's not
        missing_vars = []
        if not self.REMOTE_USER:
            missing_vars.append(self.REMOTE_USER_ENV_VAR)
        if not self.REMOTE_HOST:
            missing_vars.append(self.REMOTE_HOST_ENV_VAR)
        if not self.REMOTE_DIR:
            missing_vars.append(self.REMOTE_DIR_ENV_VAR)

        for var in missing_vars:
            self.logger.warning(f"Environment variable {var} is not set.")

        if not all([self.REMOTE_USER, self.REMOTE_HOST, self.REMOTE_DIR]):
            self.logger.warning("Disabling remote sync due to incomplete environment configuration.")
            self.REMOTE_SYNC_ENABLED = False

        # If USE_PARAMIKO is enabled but paramiko is not available, disable remote sync
        if self.REMOTE_SYNC_ENABLED and self.USE_PARAMIKO and not PARAMIKO_AVAILABLE:
            self.logger.error("Paramiko module is not installed. Remote sync requires paramiko.")
            self.REMOTE_SYNC_ENABLED = False

        # If remote sync is enabled, but still incomplete configuration, disable
        if self.REMOTE_SYNC_ENABLED:
            if self.USE_PARAMIKO:
                if not self.REMOTE_SSH_KEY:
                    self.logger.warning("Remote sync is enabled with paramiko, but SSH key is not set.")
                    self.REMOTE_SYNC_ENABLED = False
            else:
                # For system SSH, ensure that SSH key is available
                if not self.REMOTE_SSH_KEY:
                    self.logger.warning("Remote sync is enabled with system SSH, but SSH key is not set.")
                    self.REMOTE_SYNC_ENABLED = False

        if self.REMOTE_SYNC_ENABLED:
            self.logger.info("Remote sync is enabled.")
            if self.USE_PARAMIKO:
                self.logger.info("Remote sync method: Paramiko (SFTP).")
            else:
                self.logger.info("Remote sync method: System SSH (scp).")
        else:
            self.logger.info("Remote sync is disabled.")

        # Ensure that the remote_sync logger propagates to the root logger
        self.logger.propagate = True

        # Perform one-time verification and creation of remote directory
        if self.REMOTE_SYNC_ENABLED:
            success = self._verify_and_create_remote_directory()
            if not success:
                self.logger.error("Failed to verify or create the remote directory. Disabling remote sync.")
                self.REMOTE_SYNC_ENABLED = False

    def _verify_and_create_remote_directory(self):
        """
        Verifies the existence of the remote directory.
        Attempts to create it if it doesn't exist.
        Returns True if the directory exists or was created successfully, False otherwise.
        """
        try:
            if self.USE_PARAMIKO:
                self.logger.info("Verifying remote directory using Paramiko (SFTP).")
                transport = paramiko.Transport((self.REMOTE_HOST, 22))
                try:
                    if self.REMOTE_SSH_KEY:
                        ssh_key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(self.REMOTE_SSH_KEY))
                    else:
                        ssh_key = paramiko.RSAKey.from_private_key_file(os.path.expanduser('~/.ssh/id_rsa'))
                except FileNotFoundError:
                    self.logger.error("SSH key file not found. Cannot verify/create remote directory.")
                    return False

                transport.connect(username=self.REMOTE_USER, pkey=ssh_key)
                sftp = paramiko.SFTPClient.from_transport(transport)

                remote_dir = self.REMOTE_DIR
                self.mkdir_p_sftp(sftp, remote_dir)

                transport.close()
                self.logger.info(f"Remote directory '{remote_dir}' verified/created successfully.")
                return True
            else:
                self.logger.info("Verifying remote directory using system SSH (scp).")
                remote_user_host = f"{self.REMOTE_USER}@{self.REMOTE_HOST}"
                ssh_key = self.REMOTE_SSH_KEY or os.path.expanduser("~/.ssh/id_rsa")

                # Execute mkdir -p on the remote server
                ssh_command = [
                    "ssh",
                    "-i", ssh_key,
                    remote_user_host,
                    f"mkdir -p {self.REMOTE_DIR}"
                ]
                self.logger.info(f"Ensuring remote directory exists: {self.REMOTE_DIR}")
                result = subprocess.run(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode != 0:
                    self.logger.error(f"Failed to create remote directory '{self.REMOTE_DIR}' via SSH: {result.stderr.decode().strip()}")
                    return False
                else:
                    self.logger.info(f"Remote directory '{self.REMOTE_DIR}' verified/created successfully.")
                    return True
        except Exception as e:
            self.logger.error(f"Exception during remote directory verification/creation: {e}")
            return False

    def start(self):
        """
        Starts the remote sync thread.
        """
        if self.REMOTE_SYNC_ENABLED:
            self.remote_sync_thread = threading.Thread(
                target=self.remote_sync_thread_function,
                args=(self.remote_sync_queue, self.remote_sync_stop_event),
                daemon=True  # Ensures thread exits when main program exits
            )
            self.remote_sync_thread.start()
            self.logger.info("Remote sync thread started.")

    def stop(self):
        """
        Stops the remote sync thread gracefully.
        """
        if self.REMOTE_SYNC_ENABLED and self.remote_sync_thread is not None:
            self.remote_sync_stop_event.set()
            self.remote_sync_thread.join()
            self.logger.info("Remote sync thread stopped.")

    def sync_file_to_remote(self, file_path):
        """
        Syncs a given file to the remote server using the configured method with retries.

        :param file_path: Path to the file to be synced.
        """
        attempt = 0
        while attempt < self.MAX_RETRIES:
            try:
                if self.USE_PARAMIKO:
                    self.logger.info("Using Paramiko (SFTP) for syncing.")
                    self._sync_with_paramiko(file_path)
                else:
                    self.logger.info("Using system SSH (scp) for syncing.")
                    self._sync_with_system_ssh(file_path)
                # If sync is successful, break out of the loop
                break
            except Exception as e:
                attempt += 1
                self.logger.error(f"Attempt {attempt} failed to sync {file_path}: {e}")
                if attempt < self.MAX_RETRIES:
                    self.logger.info(f"Retrying in {self.RETRY_DELAY} seconds...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    self.logger.error(f"All {self.MAX_RETRIES} attempts failed to sync {file_path}.")

    def _sync_with_paramiko(self, file_path):
        """
        Syncs a file to the remote server using paramiko.

        :param file_path: Path to the file to be synced.
        """
        transport = None
        try:
            transport = paramiko.Transport((self.REMOTE_HOST, 22))
            # Authentication
            try:
                if self.REMOTE_SSH_KEY:
                    ssh_key = paramiko.RSAKey.from_private_key_file(os.path.expanduser(self.REMOTE_SSH_KEY))
                else:
                    ssh_key = paramiko.RSAKey.from_private_key_file(os.path.expanduser('~/.ssh/id_rsa'))
            except FileNotFoundError:
                self.logger.error("SSH key file not found. Remote sync cannot proceed.")
                return

            transport.connect(username=self.REMOTE_USER, pkey=ssh_key)
            sftp = paramiko.SFTPClient.from_transport(transport)

            remote_path = os.path.join(self.REMOTE_DIR, os.path.relpath(file_path, self.save_dir_base))
            sftp.put(file_path, remote_path)

            self.logger.info(f"Successfully synced {file_path} to {self.REMOTE_HOST}:{remote_path}")
        except Exception as e:
            self.logger.error(f"Failed to sync {file_path} to remote server with paramiko: {e}")
            raise  # Re-raise exception to handle retries
        finally:
            if transport:
                transport.close()

    def _sync_with_system_ssh(self, file_path):
        """
        Syncs a file to the remote server using the system's scp command.

        :param file_path: Path to the file to be synced.
        """
        try:
            remote_path = os.path.join(self.REMOTE_DIR, os.path.relpath(file_path, self.save_dir_base))
            ssh_key = self.REMOTE_SSH_KEY or os.path.expanduser("~/.ssh/id_rsa")
            remote_user_host = f"{self.REMOTE_USER}@{self.REMOTE_HOST}"
            
            # Perform SCP to transfer the file
            scp_command = [
                "scp",
                "-i", ssh_key,
                file_path,
                f"{remote_user_host}:{remote_path}"
            ]
            self.logger.info(f"Starting SCP transfer for file: {file_path} to {remote_path}")
            subprocess.run(scp_command, check=True)
            self.logger.info(f"Successfully synced {file_path} using system SSH (scp).")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to sync {file_path} using system SSH (scp): {e}")
            raise  # Re-raise exception to handle retries
        except Exception as e:
            self.logger.error(f"Unexpected error during system SSH (scp) sync of {file_path}: {e}")
            raise  # Re-raise exception to handle retries

    def mkdir_p_sftp(self, sftp, remote_directory):
        """
        Ensures that the remote directory exists. Creates it if it doesn't.

        :param sftp: The SFTP client instance.
        :param remote_directory: The remote directory path.
        """
        dirs = []
        while len(remote_directory) > 1:
            dirs.append(remote_directory)
            remote_directory, _ = os.path.split(remote_directory)
        dirs.reverse()
        for dir in dirs:
            try:
                sftp.stat(dir)
            except FileNotFoundError:
                try:
                    sftp.mkdir(dir)
                    self.logger.info(f"Created remote directory: {dir}")
                except Exception as e:
                    self.logger.error(f"Failed to create remote directory {dir}: {e}")
                    raise  # Re-raise to handle synchronization failure

    def remote_sync_thread_function(self, sync_queue, stop_event):
        """
        The main function for the remote sync thread. Processes files from the queue.

        :param sync_queue: Queue containing file paths to sync.
        :param stop_event: Event to signal the thread to stop.
        """
        last_aggregated_detections_mtime = None

        while not stop_event.is_set():
            # Handle files in the sync_queue
            try:
                file_to_sync = sync_queue.get(timeout=1)
                # Perform the sync operation with retries
                self.sync_file_to_remote(file_to_sync)
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in remote_sync_thread: {e}")

            # Check for aggregated detections file
            if self.SYNC_AGGREGATED_DETECTIONS and self.aggregated_detections_file:
                try:
                    mtime = os.path.getmtime(self.aggregated_detections_file)
                    if last_aggregated_detections_mtime is None or mtime > last_aggregated_detections_mtime:
                        # File has been modified, sync it with retries
                        self.sync_file_to_remote(self.aggregated_detections_file)
                        last_aggregated_detections_mtime = mtime
                except FileNotFoundError:
                    pass
                except Exception as e:
                    self.logger.error(f"Error checking aggregated detections file: {e}")

            # Sleep before next check
            time.sleep(1)

    def enqueue_file(self, file_path):
        """
        Adds a file path to the sync queue.

        :param file_path: Path to the file to be synced.
        """
        if self.REMOTE_SYNC_ENABLED:
            try:
                self.remote_sync_queue.put(file_path, timeout=5)
                self.logger.debug(f"Enqueued file for remote sync: {file_path}")
            except Queue.Full:
                self.logger.error(f"Remote sync queue is full. Failed to enqueue file: {file_path}")

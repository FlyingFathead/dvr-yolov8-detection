# remote_sync.py

import os
import subprocess
import logging
import threading
import time
from queue import Queue, Empty, Full
from concurrent.futures import ThreadPoolExecutor
from configparser import NoOptionError

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
        self.STRIP_LOCAL_PATH = config.getboolean('remote_sync', 'strip_local_path', fallback=True)        
        self.SYNC_AGGREGATED_DETECTIONS = config.getboolean('remote_sync', 'sync_aggregated_detections', fallback=True)
        self.SYNC_DETECTION_AREA_IMAGES = config.getboolean('remote_sync', 'sync_detection_area_images', fallback=True)
        self.SYNC_FULL_FRAME_IMAGES = config.getboolean('remote_sync', 'sync_full_frame_images', fallback=False)

        # Retry Configuration
        self.MAX_RETRIES = config.getint('remote_sync', 'max_retries', fallback=3)
        self.RETRY_DELAY = config.getint('remote_sync', 'retry_delay', fallback=5)

        # **Batch Processing Configuration**
        self.BATCH_SIZE = config.getint('remote_sync', 'batch_size', fallback=10)
        self.BATCH_TIME = config.getint('remote_sync', 'batch_time', fallback=5)

        self.logger.info(f"Batch processing enabled with batch size: {self.BATCH_SIZE} and batch time: {self.BATCH_TIME} seconds.")

        # Queue Configuration
        self.REMOTE_SYNC_QUEUE_MAXSIZE = config.getint('remote_sync', 'remote_sync_queue_maxsize', fallback=1000)
        self.remote_sync_queue = Queue(maxsize=self.REMOTE_SYNC_QUEUE_MAXSIZE)
        self.remote_sync_stop_event = threading.Event()
        self.remote_sync_thread = None

        # Batch Interval Configuration
        self.BATCH_INTERVAL = config.getfloat('remote_sync', 'batch_interval', fallback=0)
        self.logger.info(f"[NOT IN USE] Remote sync batch interval set to {self.BATCH_INTERVAL} seconds.")

        # handle via thread pool executor        
        # self.executor = ThreadPoolExecutor(max_workers=5)  # Adjust the number of workers as needed

        # determine maximum number of workers for remote sync
        # Read max_workers from config with a default fallback
        try:
            self.MAX_WORKERS = config.getint('remote_sync', 'max_workers')
        except (NoOptionError, ValueError):
            self.MAX_WORKERS = 5  # Default value if not specified or invalid
        self.logger.info(f"Remote sync will use up to {self.MAX_WORKERS} worker threads.")

        # Ensure MAX_WORKERS is a positive integer
        if self.MAX_WORKERS <= 0:
            self.logger.warning("max_workers must be a positive integer. Defaulting to 5.")
            self.MAX_WORKERS = 5

        # Initialize the ThreadPoolExecutor with the configured number of workers
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)

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

        # Retrieve SSH and SCP paths from environment variables
        self.SSH_BIN = os.getenv('SSH_BIN') if not self.USE_PARAMIKO else None
        self.SCP_BIN = os.getenv('SCP_BIN') if not self.USE_PARAMIKO else None

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

        # If remote sync is enabled, handle SSH key requirements
        if self.REMOTE_SYNC_ENABLED:
            if self.USE_PARAMIKO:
                if not self.REMOTE_SSH_KEY:
                    self.logger.warning("Remote sync is enabled with paramiko, but SSH key is not set.")
                    self.REMOTE_SYNC_ENABLED = False
            else:
                # For system SSH, SSH key is optional. Use default if not set.
                if not self.REMOTE_SSH_KEY:
                    self.logger.info("Remote sync is enabled with system SSH, using default SSH key (~/.ssh/id_rsa).")
                else:
                    self.logger.info(f"Remote sync is enabled with system SSH, using SSH key: {self.REMOTE_SSH_KEY}")

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

                # Execute mkdir -p on the remote server using absolute ssh path
                ssh_command = [
                    self.SSH_BIN,
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

        self.logger.info(f"Starting sync for {file_path} in thread {threading.current_thread().name}")

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
                raise

            transport.connect(username=self.REMOTE_USER, pkey=ssh_key)
            sftp = paramiko.SFTPClient.from_transport(transport)

            # Determine remote path based on STRIP_LOCAL_PATH
            if self.STRIP_LOCAL_PATH:
                remote_path = os.path.join(self.REMOTE_DIR, os.path.basename(file_path))
            else:
                # Preserve the relative directory structure
                relative_path = os.path.relpath(file_path, self.save_dir_base)
                remote_path = os.path.join(self.REMOTE_DIR, relative_path)

            # Ensure the remote directory exists
            remote_dir = os.path.dirname(remote_path)
            self.mkdir_p_sftp(sftp, remote_dir)

            # Upload the file
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
            if self.STRIP_LOCAL_PATH:
                remote_path = os.path.join(self.REMOTE_DIR, os.path.basename(file_path))
            else:
                # Preserve the relative directory structure
                relative_path = os.path.relpath(file_path, self.save_dir_base)
                remote_path = os.path.join(self.REMOTE_DIR, relative_path)

            ssh_key = self.REMOTE_SSH_KEY or os.path.expanduser("~/.ssh/id_rsa")
            remote_user_host = f"{self.REMOTE_USER}@{self.REMOTE_HOST}"

            # Perform SCP to transfer the file using absolute path to scp
            scp_command = [
                self.SCP_BIN,
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
        last_queue_log_time = 0  # Initialize last log time        

        while not stop_event.is_set():
            current_time = time.time()
            # Log queue size every 10 seconds
            if current_time - last_queue_log_time >= 10:
                queue_size = sync_queue.qsize()
                self.logger.info(f"Remote sync queue size: {queue_size}/{self.REMOTE_SYNC_QUEUE_MAXSIZE}")
                last_queue_log_time = current_time

            # Existing warning if queue size is high
            queue_size = sync_queue.qsize()
            if queue_size > self.REMOTE_SYNC_QUEUE_MAXSIZE * 0.8:
                self.logger.warning(f"Remote sync queue size is at {queue_size}/{self.REMOTE_SYNC_QUEUE_MAXSIZE}")

            # Handle files in the sync_queue
            try:
                file_to_sync = sync_queue.get(timeout=1)
                queue_size = sync_queue.qsize()
                self.logger.info(f"Dequeued file for remote sync: {file_to_sync} (Queue size: {queue_size}/{self.REMOTE_SYNC_QUEUE_MAXSIZE})")
                self.executor.submit(self.sync_file_to_remote, file_to_sync)
            except Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error in remote_sync_thread: {e}")

        # while not stop_event.is_set():
        #     # Handle files in the sync_queue
        #     queue_size = sync_queue.qsize()
        #     if queue_size > self.REMOTE_SYNC_QUEUE_MAXSIZE * 0.8:
        #         self.logger.warning(f"Remote sync queue size is at {queue_size}/{self.REMOTE_SYNC_QUEUE_MAXSIZE}")

        #     # Handle files in the sync_queue
        #     try:
        #         file_to_sync = sync_queue.get(timeout=1)
        #         queue_size = sync_queue.qsize()
        #         self.logger.info(f"Dequeued file for remote sync: {file_to_sync} (Queue size: {queue_size}/{self.REMOTE_SYNC_QUEUE_MAXSIZE})")
        #         # Perform the sync operation with retries
        #         self.executor.submit(self.sync_file_to_remote, file_to_sync)
        #     except Empty:
        #         pass
        #     except Exception as e:
        #         self.logger.error(f"Error in remote_sync_thread: {e}")

            # # Handle files in the sync_queue
            # try:
            #     file_to_sync = sync_queue.get(timeout=1)
            #     # Perform the sync operation with retries
            #     # self.sync_file_to_remote(file_to_sync)
            #     self.executor.submit(self.sync_file_to_remote, file_to_sync)                
            # except Empty:
            #     pass
            # except Exception as e:
            #     self.logger.error(f"Error in remote_sync_thread: {e}")

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
                self.remote_sync_queue.put(file_path, block=False)
                queue_size = self.remote_sync_queue.qsize()
                self.logger.info(f"Enqueued file for remote sync: {file_path} (Queue size: {queue_size}/{self.REMOTE_SYNC_QUEUE_MAXSIZE})")
            except Full:
                self.logger.error(f"Remote sync queue is full. Failed to enqueue file: {file_path}")
                raise  # Re-raise to be caught by the caller

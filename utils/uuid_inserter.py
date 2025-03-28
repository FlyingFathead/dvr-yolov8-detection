# ./utils/uuid_inserter.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper utility to add missing UUIDs to an aggregated_detections.json file.
# Creates a timestamped backup before modification.
# Reads default path from config.ini (relative to project root).
# Allows specifying a different file via command line.
# https://github.com/FlyingFathead/dvr-yolov8-detection/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import json
import uuid
import os
import shutil
import logging
import configparser
import argparse
from datetime import datetime

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('uuid_inserter')

# --- Path Configuration (Relative to this script's location) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # Assumes utils/ is one level down from root
CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config.ini')

# --- Load Config Function ---
def load_config(config_path=CONFIG_FILE_PATH):
    """Loads configuration from the specified INI file."""
    config = configparser.ConfigParser(interpolation=None)
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        return None # Return None if config not found

    try:
        read_files = config.read(config_path)
        if not read_files:
            logger.error(f"Configuration file '{config_path}' is empty.")
            return None
        return config
    except configparser.Error as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}")
        return None


# --- Main Processing Function ---
def add_missing_uuids(target_file_path):
    """
    Reads the specified JSON file, adds UUIDs to entries missing them,
    and writes back to the *same file*. Creates a backup before modifying.
    Handles paths correctly relative to the project root or absolute paths.
    """
    # Ensure the target path is absolute for clarity
    abs_target_path = os.path.abspath(target_file_path)
    logger.info(f"Attempting to process file: {abs_target_path}")

    # 1. Check if file exists
    if not os.path.exists(abs_target_path):
        logger.error(f"Target file not found: {abs_target_path}")
        return False # Indicate failure

    # 2. Safety First: Backup the file
    backup_file_path = f"{abs_target_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
    try:
        shutil.copy2(abs_target_path, backup_file_path) # copy2 preserves metadata
        logger.info(f"Backup created: {backup_file_path}")
    except Exception as e:
        logger.error(f"Failed to create backup file: {e}")
        logger.error("Aborting process to prevent data loss.")
        return False # Indicate failure

    # 3. Read the JSON data
    detection_data = []
    try:
        with open(abs_target_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip(): # Check if file is empty or just whitespace
                logger.warning(f"File '{abs_target_path}' is empty. Nothing to process.")
                # No need to write back, consider it a success (no data loss)
                return True
            # Reset file pointer and load JSON
            f.seek(0)
            detection_data = json.load(f)
            if not isinstance(detection_data, list):
                 logger.error(f"Expected a JSON list in {abs_target_path}, but got {type(detection_data)}. Aborting.")
                 logger.error(f"Restore data from backup: {backup_file_path}")
                 return False
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {abs_target_path}: {e}")
        logger.error("Please check the file for formatting errors.")
        logger.error(f"Restore data from backup: {backup_file_path}")
        return False
    except Exception as e:
        logger.error(f"Error reading file {abs_target_path}: {e}")
        logger.error(f"Restore data from backup: {backup_file_path}")
        return False

    # 4. Process the data: Add missing UUIDs
    added_count = 0
    existing_count = 0
    processed_count = 0
    skipped_malformed = 0
    modified_data = [] # Use a new list to store processed/valid entries

    for i, entry in enumerate(detection_data):
        processed_count += 1
        if not isinstance(entry, dict):
            logger.warning(f"Skipping entry #{i+1}: Expected a dictionary, but got {type(entry)}")
            skipped_malformed += 1
            continue # Skip this entry entirely

        existing_uuid = entry.get('uuid')
        # Check for existing, non-empty, valid UUID-like string
        is_valid_existing = False
        if existing_uuid and isinstance(existing_uuid, str) and len(existing_uuid.strip()) > 0:
             try:
                 # Try parsing it to see if it's a valid UUID format
                 uuid.UUID(existing_uuid)
                 is_valid_existing = True
             except ValueError:
                 logger.warning(f"Entry #{i+1}: Found existing 'uuid' but it's not a valid format: '{existing_uuid}'. Will overwrite.")
                 is_valid_existing = False # Treat as missing/invalid

        if is_valid_existing:
            existing_count += 1
            modified_data.append(entry) # Keep valid existing entry
        else:
            new_uuid = str(uuid.uuid4())
            entry['uuid'] = new_uuid # Add or overwrite UUID
            added_count += 1
            logger.info(f"Entry #{i+1}: Added/updated UUID: {new_uuid}")
            modified_data.append(entry) # Add modified entry

    logger.info("-" * 30)
    logger.info(f"Processing summary for: {abs_target_path}")
    logger.info(f"  Total entries read:      {len(detection_data)}")
    # logger.info(f"  Entries processed:       {processed_count}") # Less useful than read/written
    logger.info(f"  Skipped (malformed):   {skipped_malformed}")
    logger.info(f"  UUIDs added/overwritten: {added_count}")
    logger.info(f"  Valid existing UUIDs:  {existing_count}")
    logger.info(f"  Entries to write back: {len(modified_data)}")
    logger.info("-" * 30)


    # 5. Write the modified data back (only if changes were made or entries skipped)
    if added_count > 0 or skipped_malformed > 0:
        logger.info(f"Writing {len(modified_data)} processed entries back to {abs_target_path}...")
        try:
            with open(abs_target_path, 'w', encoding='utf-8') as f:
                # Use indent for readability, default=str for safety (though shouldn't be needed now)
                json.dump(modified_data, f, indent=4, default=str)
            logger.info("File successfully updated.")
            return True # Indicate success
        except Exception as e:
            logger.error(f"Error writing updated data to {abs_target_path}: {e}")
            logger.error(f"Your original data is safe in the backup: {backup_file_path}")
            return False # Indicate failure
    else:
        logger.info("No missing/invalid UUIDs found and no entries skipped. File remains unchanged.")
        return True # Indicate success (no changes needed)

# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting UUID insertion utility...")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Add missing UUIDs to aggregated detection JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to the JSON file to process. If not provided, uses the path from config.ini.",
        type=str,
        default=None # Default is None, indicating we should use config
    )
    args = parser.parse_args()

    # --- Determine Target File Path ---
    target_file = None
    if args.file:
        # User provided a specific file path
        target_file = args.file
        logger.info(f"Using specified file from command line: {target_file}")
    else:
        # No file specified, try loading from config
        logger.info(f"No file specified via command line, attempting to load from {CONFIG_FILE_PATH}")
        config = load_config()
        if config:
            try:
                default_path_relative = config.get(
                    'aggregation',
                    'aggregated_detections_file'
                    # No fallback needed here, we need the key to exist if using config
                )
                # Assume path in config is relative to PROJECT_ROOT
                target_file = os.path.join(PROJECT_ROOT, default_path_relative)
                logger.info(f"Using default file path from config.ini: {target_file} (resolved from '{default_path_relative}')")
            except (configparser.NoSectionError, configparser.NoOptionError):
                logger.error("Could not find '[aggregation]' section or 'aggregated_detections_file' option in config.ini.")
                target_file = None
        else:
            # Config file itself failed to load
             target_file = None

    # --- Execute Processing ---
    if target_file:
        success = add_missing_uuids(target_file)
        if success:
             logger.info("Utility finished successfully.")
             sys.exit(0) # Exit code 0 for success
        else:
             logger.error("Utility finished with errors.")
             sys.exit(1) # Exit code 1 for failure
    else:
        logger.error("Could not determine target file path. Please specify with --file or ensure config.ini is correct.")
        sys.exit(1) # Exit code 1 for failure
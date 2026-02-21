from datetime import datetime
import os


class FireBotLogger:
    def __init__(self, base_dir: str = "logs", experiment_name: str = "default") -> None:

        # Build the log directory name
        iso_time: str = datetime.now().isoformat()
        folder_name: str = f"{iso_time}_{experiment_name}"

        # Save the absolute path
        self.log_dir: str = os.path.abspath(os.path.join(base_dir, folder_name))

        # Create the log directory
        os.makedirs(self.log_dir, exist_ok=False)
        print(f"Created log directory: {self.log_dir}")

    def get_log_dir(self) -> str:
        return self.log_dir

if __name__ == "__main__":
    logger = FireBotLogger()

    print(logger.get_log_dir())

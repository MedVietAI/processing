# Check Google Drive status
from utils.drive_saver import DriveSaver

if __name__ == "__main__":
    ds = DriveSaver()
    if ds.is_service_available():
        print("Drive ready.")
    else:
        print("Drive NOT ready.")

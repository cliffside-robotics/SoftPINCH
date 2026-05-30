from pathlib import Path

def create_recording_folder(SUBJECT_NAME, BASE_PATH):
    """
    Create a recording folder for a given subject inside BASE_PATH.
    Subfolders: EEG, EMG, MC, Markers

    Parameters
    ----------
    SUBJECT_NAME : str
        Name of the subject folder to create.
    BASE_PATH : str or Path
        Base directory where the subject folder will be created.

    Returns
    -------
    dict
        A dictionary containing the created folder paths.
    """
    # Ensure BASE_PATH is a Path object
    base_path = Path(BASE_PATH)
    subject_folder = base_path / SUBJECT_NAME

    # Create subject folder and subfolders
    subfolders = ["EEG", "EMG", "MC", "Markers"]
    for folder in subfolders:
        (subject_folder / folder).mkdir(parents=True, exist_ok=True)

    print(f"[OK] Created folder structure for subject: {SUBJECT_NAME}")

    # Return dictionary of paths
    return {
        "BASE": subject_folder,
        "EEG": subject_folder / "EEG",
        "EMG": subject_folder / "EMG",
        "MC": subject_folder / "MC",
        "Markers": subject_folder / "Markers",
    }

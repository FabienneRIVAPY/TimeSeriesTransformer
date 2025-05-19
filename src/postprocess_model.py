def save_path_to_file(path_str, output_filename):
    """
    Write a path string directly to a specified output file.

    Args:
        path_str (str): The path string to be saved
        output_filename (str): The filename where the path will be written

    Returns:
        None
    """
    # Write to file directly
    with open(output_filename, "w") as f:
        f.write(path_str)

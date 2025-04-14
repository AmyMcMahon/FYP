import json
import zstandard
import sys
import os

# Function to read lines from .zst file
def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = reader.read(2**27).decode('utf-8', errors='ignore')
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line
            buffer = lines[-1]
        reader.close()

# Function to process the file
def process_file(file_path, file_type, filtered):
    subreddit_name = os.path.basename(file_path).split("_")[0]  # Extract subreddit name
    output_file = f"datasets/{subreddit_name}_{file_type}_{'filtered' if filtered else 'unfiltered'}.json"

    filtered_keys = {
        "submission": ["name", "subreddit", "title", "selftext", "score", "created_utc"],
        "comment": ["parent_id", "subreddit", "body", "score", "created_utc"]
    }

    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in read_lines_zst(file_path):
            try:
                data = json.loads(line)
                if filtered and file_type in filtered_keys:
                    data = {key: data[key] for key in filtered_keys[file_type] if key in data}
                json.dump(data, out_f)
                out_f.write("\n")
            except json.JSONDecodeError:
                continue  # Skip corrupted lines

    print(f"Processed file saved as: {output_file}")

# Main function to handle arguments
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python data_cleaning.py <file_path> <submission/comment> <filtered/unfiltered>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_type = sys.argv[2].lower()
    filtered = sys.argv[3].lower() == "filtered"

    if file_type not in ["submission", "comment"]:
        print("Error: file_type must be 'submission' or 'comment'")
        sys.exit(1)

    process_file(file_path, file_type, filtered)

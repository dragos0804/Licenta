import streamlit as st
import shutil
import os
import subprocess

vizualizer_dir = os.getcwd()
# Directories for RNN
RNN_dir = os.path.join(os.getcwd(), 'RNN')
RNN_data_dir = os.path.join(RNN_dir, 'data')
RNN_src_dir = os.path.join(RNN_dir, 'src')
files_to_be_compressed_dir = os.path.join(RNN_data_dir, 'files_to_be_compressed')
processed_files_dir = os.path.join(RNN_data_dir, 'processed_files')
RNN_compressed_dir = os.path.join(RNN_data_dir, 'compressed')
RNN_decompressed_dir = os.path.join(RNN_data_dir, 'decompressed')

# Directories for Transformer
TR_dir = os.path.join(os.getcwd(), "Transformer")
TR_data_dir = os.path.join(TR_dir, 'data')
TR_src_dir = os.path.join(TR_dir, 'src')
TR_files_to_be_compressed_dir = os.path.join(TR_data_dir, 'files_to_be_compressed')
TR_compressed_dir = os.path.join(TR_data_dir, 'compressed')
TR_decompressed_dir = os.path.join(TR_data_dir, 'decompressed')

def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                st.error(f"Failed to delete {file_path}: {e}")

def save_file_to_directory(directory, file):
    file_path = os.path.join(directory, file.name)
    with open(file_path, 'wb') as f:
        f.write(file.getbuffer())


# Function to handle archiving logic
def handle_archive(file_to_archive, archive_option):
    try:
        if archive_option == "RNN-based compression":
            # Delete everything from files_to_be_compressed directory
            st.write("Clearing files...")
            clear_directory(files_to_be_compressed_dir)

            # Delete everything from processed_files
            st.write("Clearing files...")
            clear_directory(processed_files_dir)

            # Save uploaded file to files_to_be_compressed directory
            st.write("Saving ...")
            save_file_to_directory(files_to_be_compressed_dir, file_to_archive)


            st.write("Running parser...")
            os.chdir(RNN_data_dir)
            parser_script_path = os.path.join(RNN_data_dir, 'run_parser.bat')
            if os.path.exists(parser_script_path):
                result = subprocess.run(['cmd', '/c', parser_script_path],
                                        capture_output=True, text=True)
                st.text(result.stdout)
                if result.returncode != 0:
                    st.error(result.stderr)
                    return  # Exit if parser script fails
            else:
                st.error(f"Parser script not found: {parser_script_path}")
                return  # Exit if parser script not found

            st.write("Running compression...")
            
            os.chdir(RNN_src_dir)
            compression_script_path = os.path.join(RNN_src_dir, 'compress.bat')
            if os.path.exists(compression_script_path):
                result = subprocess.run(['cmd', '/c', compression_script_path, 'biLSTM'],
                                        check=True, capture_output=True, text=True)
                st.text(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
            else:
                st.error(f"Compression script not found: {compression_script_path}")
            

            # Delete everything from files_to_be_compressed directory
            st.write("Clearing files...")
            clear_directory(files_to_be_compressed_dir)
            
            os.chdir(vizualizer_dir)
            st.success("File has been processed successfully.")
        else:
            # Delete everything from files_to_be_compressed directory
            st.write("Clearing files...")
            clear_directory(TR_files_to_be_compressed_dir)

            # Save uploaded file to files_to_be_compressed directory
            st.write("Saving ...")
            save_file_to_directory(TR_files_to_be_compressed_dir, file_to_archive)

            st.write("Running compression...")
            os.chdir(TR_src_dir)
            compression_script_path = os.path.join(TR_src_dir, 'compress.bat')
            if os.path.exists(compression_script_path):
                result = subprocess.run(['cmd', '/c', compression_script_path],
                                        check=True, capture_output=True, text=True)
                st.text(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
            else:
                st.error(f"Compression script not found: {compression_script_path}")

            # Delete everything from files_to_be_compressed directory
            st.write("Clearing files...")
            clear_directory(TR_files_to_be_compressed_dir)
            
            os.chdir(vizualizer_dir)
            st.success("File has been processed successfully.")            
    except subprocess.CalledProcessError as e:
        st.error(f"Subprocess error: {e.stdout}\n{e.stderr}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# Function to handle dearchiving logic
def handle_dearchive(file_to_dearchive, params_to_dearchive):
    # File uploader for the file to dearchive
    print(params_to_dearchive)

    prefix_basename = params_to_dearchive.name
    # Remove the file extension
    prefix_name_combined, prefix_type = os.path.splitext(prefix_basename)
    prefix_name, _ = os.path.splitext(prefix_name_combined)
    print(prefix_name)
    if file_to_dearchive and params_to_dearchive:
        if prefix_type == ".params": 
            st.write(f"You have selected to dearchive the file: {file_to_dearchive.name}")
            # Add your dearchiving logic here
            os.chdir(RNN_src_dir)
            decompression_script_path = os.path.join(RNN_src_dir, 'decompress.bat')
            if os.path.exists(decompression_script_path):
                result = subprocess.run(['cmd', '/c', decompression_script_path, prefix_name],
                                        check=True, capture_output=True, text=True)
                st.text(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
            else:
                st.error(f"Compression script not found: {decompression_script_path}")
            
            
        else:
            st.write(f"You have selected to dearchive the file: {file_to_dearchive.name}")
            # Add your dearchiving logic here
            os.chdir(TR_src_dir)
            decompression_script_path = os.path.join(TR_src_dir, 'decompress.bat')
            if os.path.exists(decompression_script_path):
                result = subprocess.run(['cmd', '/c', decompression_script_path],
                                        check=True, capture_output=True, text=True)
                st.text(result.stdout)
                if result.stderr:
                    st.error(result.stderr)
            else:
                st.error(f"Compression script not found: {decompression_script_path}")
        os.chdir(vizualizer_dir)
        st.success("File has been dearchived successfully.")

def main():
    st.title('ZipAI')

    st.write('Welcome to ZipAI, a tool dedicated for lossless compression using AI!')


    # Radio button for choosing between archiving and dearchiving
    action = st.radio("Choose an action:", ["Archive", "Dearchive"])
    
    # Conditionally display components based on the selected action
    if action == "Archive":
        archive_option = st.selectbox("Choose an archive option:", ["RNN-based compression", "Transformer-based compression"])
        file_to_archive = st.file_uploader("Upload the file to archive:", type=["txt", "pdf", "jpg", "png", "zip"])
        
        if file_to_archive:
            if st.button("Start Compressing"):
                handle_archive(file_to_archive, archive_option)
    elif action == "Dearchive":
            file_to_dearchive = st.file_uploader("Upload \".combined\" file to dearchive:", type=["combined"])
            params_to_dearchive = st.file_uploader("Upload \".json\" file to dearchive:", type=["params", "json"])

            if st.button("Start Decompressing"):
                handle_dearchive(file_to_dearchive, params_to_dearchive)

if __name__ == "__main__":
    main()
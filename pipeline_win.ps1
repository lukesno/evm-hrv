param (
    [string]$input_file,
    [string]$output_dir
)

# Set the preference to stop on errors
$ErrorActionPreference = "Stop"

Write-Host "Starting video processing script..."

try {
    # Create a new directory under output_dir using the name of the mp4 file
    $file_name = [System.IO.Path]::GetFileNameWithoutExtension($input_file)
    $final_dir = Join-Path -Path $output_dir -ChildPath $file_name

    # Check if the directory already exists
    # if (Test-Path -Path $final_dir -PathType Container) {
    #     Write-Host "###########################################################################################"
    #     Write-Host "Directory '$final_dir' already exists. Rename something, or remove the existing directory`n"
    #     Write-Host "Powershell:"
    #     Write-Host "    Remove-Item -Path 'C:\Path\To\Your\Directory' -Recurse"
    #     Write-Host "###########################################################################################"
    #     exit
    # }

    New-Item -ItemType Directory -Path $final_dir -Force | Out-Null
    Write-Host "Created directory: $final_dir"

    # Call evm.exe with the specified arguments
    Write-Host "Calling EVM" 
    Write-Host "Note: Spatial filtering can take a couple of minutes" 
    & "./EVM/EVM_Matlab_bin-1.1-win64/evm.exe" $input_file $final_dir 30 color 70/60 150/60 150 'ideal' 1 6

    # Find the first *.avi file in final_dir
    $evm_vid_path = Get-ChildItem -Path $final_dir -Filter *.avi | Select-Object -First 1 -ExpandProperty FullName
    Write-Host "Found AVI file: $evm_vid_path"

    # Call the Python script for red channel extraction
    Write-Host "Calling red channel extraction script..."
    & "python" "./opencv-red-channel-extraction/extract_red_channel.py" $evm_vid_path -o $final_dir

    # Find the first *.mp4 file in final_dir
    $channeled_smoothed_path = Get-ChildItem -Path $final_dir -Filter *.mp4 | Select-Object -First 1 -ExpandProperty FullName
    Write-Host "Found MP4 file: $channeled_smoothed_path"

    # Call the Python script for red channel over time plot
    Write-Host "Calling red channel over time plot script..."
    & "python" "./opencv-red-time-plot/plot_red_channel_over_time.py" $channeled_smoothed_path -o $final_dir

    # Find the first *.npz file in final_dir
    $npz_file_path = Get-ChildItem -Path $final_dir -Filter *.npz | Select-Object -First 1 -ExpandProperty FullName
    Write-Host "Found NPZ file: $npz_file_path"

    # Call the Python script for peak analysis
    Write-Host "Calling peak analysis script..."
    & "python" "./opencv-peak-analysis/peak_analysis.py" $npz_file_path -o $final_dir

    # Copy the input file into the final directory
    Write-Host "Copying original file to $final_dir..."
    Copy-Item -Path $input_file -Destination $final_dir -Force

    Write-Host "Video processing completed!"

} catch {
    # Handle the error
    Write-Host "Error: $_"
}
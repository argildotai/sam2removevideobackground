# SAM2 Video Background Removal

This project uses the Segment Anything 2 (SAM2) model to remove backgrounds from videos. It's designed to run on Replicate, leveraging GPU acceleration for efficient processing.

## Features

- Removes background from input videos
- Allows custom background color selection
- Uses SAM2 for accurate object segmentation
- Supports CUDA acceleration for faster processing

## Requirements

- CUDA 11.7
- Python 3.10
- FFmpeg
- Various Python libraries (see `cog.yaml` for details)

## Usage

This project is designed to be run on Replicate. To use it:

1. Upload your video to Replicate
2. Choose a background color (default is green)
3. Run the model
4. Download the processed video

## How it works

1. The input video is split into frames
2. SAM2 is used to segment the main object in the video
3. The background is replaced with the chosen color
4. The frames are recombined into a video

## Development

To set up the development environment:

1. Clone this repository
2. Install the required dependencies as specified in `cog.yaml`
3. Download the SAM2 model weights

## Acknowledgements

This project uses the Segment Anything 2 (SAM2) model from Meta AI Research.
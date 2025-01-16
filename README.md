# SAM2 Webui - forked from https://github.com/zhaibowen/sam2-video-webui

## Usage on CRC machines

### Setup

1. Connection to SAM2 repo
	a. Modify line 1 of `paths.txt` with the path to your [SAM2](https://github.com/facebookresearch/sam2) repository.
2. (If necessary) Install gradio and nodejs in conda environment
	a. Activate sam2 conda environment
	b. run `pip install gradio` and test its installation with `conda list gradio`
	c. run `conda install -c conda-forge nodejs`

3. Supply data 
	a. Modify line 2 of `paths.txt` with the path to your data directory (numbered .jpg images representing a video)

4. Set preferences

	a. Optionally compile sam2 by changing `compile_sam=False` to `compile_sam=True` in `segment.py`. Note compiling makes loading the model, as well as its first inference, slow. Compiling only offers speedup for long videos. 

### Run

**Option 1: CRC frontend (for small jobs)**
run `python segment.py`

**Option 2: CRC backend (for large jobs)**
NOTE: As of now, to access the public link from the CRC backend, you must wait 5 minutes for a file sync (see step 2), so I recommend the frontend for small jobs.

1. run `qsub run.sh` 
2. Wait 300 seconds (5 minutes) for the backend to sync the file with the frontend. 
   This is the best solution I could find, see https://docs.crc.nd.edu/general_pages/s/synchronizing_files_afs.html?highlight=i%20o
3. Run `cat gradio_link.txt`. Open the public link in a browswer to run the gradio app.

When finished, kill the job by running qdel -j <job_id>


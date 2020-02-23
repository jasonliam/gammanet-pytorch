Usage

Made specifically for gammanet experiments, but feel free to adapt this for other purposes. 

- The scripts are designed around batch-job notebooks. Each batch job must have a corresponding notebook on persisitent storage, e.g. ceph. 
- The job yaml launches the entrypoint script upon pod start, passing an experiment code as argument. 
- The entrypoint script locates the job notebook according to the experiment code, converts and launches it. 
- The notebook is responsible for saving its results to a folder named witht the experiment code. 
- The entrypoint script would compress the results folder and save it back to ceph before exiting. 
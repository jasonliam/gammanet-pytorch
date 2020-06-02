if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment code> " >&2
  exit 1
fi
if [ "$#" == 2 ]; then
  cat batch_job.yaml | sed "s/EXP_CODE/$1/g" | sed "s/DATA_F/$2/g" | kubectl delete -f -
else 
  cat batch_job.yaml | sed "s/EXP_CODE/$1/g" | sed "s/DATA_F//g" | kubectl delete -f -
fi

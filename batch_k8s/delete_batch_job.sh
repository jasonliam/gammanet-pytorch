if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment code> " >&2
  exit 1
fi
cat batch_job.yaml | sed "s/EXP_CODE/$1/g" | kubectl delete -f -
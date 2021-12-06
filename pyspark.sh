export PYSPARK_PYTHON=/usr/bin/python2.7
export PYSPARK_DRIVER_PYTHON=/opt/{group_name}/{rd_name}/miniconda2/bin/python
# export PYSPARK_PYTHON=/opt/{group_name}/anaconda3/bin/python
export PYTHONSTARTUP=/opt/{group_name}/{rd_name}/spark/startup_config.py
# /opt/{group_name}/spark-2.2/python/pyspark/shell.py
# export PYSPARK_DRIVER_PYTHON=ipython
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --ip=0.0.0.0 --port=${port} --NotebookApp.password='' --NotebookApp.notebook_dir=./notebooks --debug --no-browser --NotebookApp.certfile='key/mycert.pem'"
#  --NotebookApp.certfile='key/mycert.pem' --NotebookApp.keyfile='key/mykey.key'

export JUPYTER_CONFIG_DIR=/opt/{group_name}/{rd_name}/jupyter/config
export JUPYTER_DATA_DIR=/opt/{group_name}/{rd_name}/jupyter/data

# --master yarn --deploy-mode client 
  # --master yarn \
  # --master local[*] \

$SPARK_HOME/bin/pyspark --queue ${queue_name} \
  --master yarn \
  --name {rd_name}_pyspark_shell \
  --driver-memory 20g \
  --executor-memory 2g \
  --num-executors 500 \
  --executor-cores 1 \
  --jars viewfs:///user/xxxx/{rd_name}/lib/spark-tensorflow-connector-1.0-SNAPSHOT.jar \
  --jars viewfs:///user/xxxx/{rd_name}/lib/tensorflow-hadoop-1.0-06262017-SNAPSHOT-shaded-protobuf.jar\
  --archives viewfs:///user/xxxx/{rd_name}/lib/python_libs.zip

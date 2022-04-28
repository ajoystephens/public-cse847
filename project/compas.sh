#!/bin/bash
image_root=img/compas/
prepared_data=prepared_data/compas.csv
clustered_data=clustered_data/compas.csv


echo "STEP 1: Getting Data and Running Classifier"
python3 0_data_prep__compas.py $image_root $prepared_data

echo " "

echo "STEP 2: Finding Least Accurate Cluster"
python3 1_cluster_data.py $image_root $prepared_data $clustered_data

echo " "

echo "STEP 3: Investigating Least Accurate Cluster"
python3 2_cluster_investigate.py $image_root $clustered_data
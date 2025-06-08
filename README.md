#### Follow the instructions below to run the entire pipeline. Below is an example to run Pipeline-1 on the L-CAS Dataset using the DCCLA Detector. You can run Pipeline-2 similarly on the JRDB Dataset. We recommend that you use Google Colab to get the GPU access easily.

#### Step 1: Clone the whole repository (you can choose later Pipeline-1 or Pipeline-2)
!git clone https://github.com/sehab1611251/3D-Group-Detection-Using-Distance-Based-Clustering.git

#### Step 2: Move into the Pipeline-1 folder 
%cd 3D-Group-Detection-Using-Distance-Based-Clustering/Pipeline-1
#### List to verify you are in the correct directory
!ls

#### Step 3: Install both Detector (RPEA and DCCLA) models so that you can use any one later (It will take some time).
#### RPEA
!bash environment_setup_rpea.sh
#### DCCLA
!bash environment_setup_dccla.sh

#### Step 4: Copy pipeline‐1 scripts into the following location to use the DCCLA Detector
#### copy detection + grouping helper into DCCLA
!cp pedestrian_detection_and_grouping_lcas.py /content/DCCLA/
#### copy evaluation & visualization into DCCLA too
!cp evaluation_on_L-CAS.py /content/DCCLA/
!cp visualization_L-CAS.py /content/DCCLA/

#### Step 5: Run detection + grouping with DCCLA
!/content/py39_env/bin/python /content/DCCLA/pedestrian_detection_and_grouping_lcas.py \
  --model dccla \
  --pcd_zip   "/content/LCAS_20160523_1200_1218_pcd.zip" \
  --labels_zip "/content/LCAS_20160523_1200_1218_labels.zip" \
  --ckpt_path  "/content/DCCLA/DCCLA_JRDB2022.pth" \
  --distance_threshold 1.5 \
  --output_dir        /content/dccla_detector_output \
  --det_output_file   /content/det.txt \
  --group_output_file /content/group_detections.txt

#### Step 6: Evaluate on L-CAS
!/content/py39_env/bin/python /content/DCCLA/evaluation_on_L-CAS.py \
  --gt_dir    "/content/dccla_detector_output/labels_unzipped/LCAS_20160523_1200_1218_labels" \
  --pred_file "/content/group_detections.txt" \
  --threshold 0.5

#### Step 7: Visualize results
!/content/py39_env/bin/python /content/DCCLA/visualization_L-CAS.py \
  --gt_folder "/content/dccla_detector_output/labels_unzipped/LCAS_20160523_1200_1218_labels" \
  --gt_txt    "/content/GT.txt" \
  --pcd_dir   "/content/dccla_detector_output/pcd_unzipped/LCAS_20160523_1200_1218_pcd" \
  --det_file  "/content/det.txt" \
  --pred_file "/content/group_detections.txt" \
  --out_gt_ind   "/content/gt_individual" \
  --out_gt_group "/content/gt_group" \
  --out_detector "/content/dccla_images" \
  --out_pred      "/content/pred_group_images"

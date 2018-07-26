#! /bin/bash
# create_vm_instance.sh

export YOUR_PRJ_NAME=dlcampjeju2018
export YOUR_ZONE=us-central1-f
export YOUR_VM=acheketa3-vm

echo  Set your proj and zone again
gcloud config set project $YOUR_PRJ_NAME
gcloud config set compute/zone $YOUR_ZONE


echo CREATE GCLOUD VM
gcloud compute instances create $YOUR_VM \
  --machine-type=n1-standard-2 \
  --image-project=ml-images \
  --image-family=tf-1-8 \
  --scopes=cloud-platform
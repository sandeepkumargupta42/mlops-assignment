## MLOps-Group21

dvc init

dvc remote add -d myremote gdrive://1N_F9Q92cwYce6cs6ztGv5EgnnemRRSGb

dvc add data

dvc remote add --default myremote gdrive://1N_F9Q92cwYce6cs6ztGv5EgnnemRRSGb
dvc remote modify myremote gdrive_acknowledge_abuse true
dvc push

dvc remote modify myremote --local gdrive_user_credentials_file secret/client_secret.json

dvc remote modify myremote gdrive_use_service_account true
dvc remote modify myremote --local gdrive_service_account_json_file_path secret/assignment_service.json


------------------

dvc pull

git init
dvc init
dvc add data/daily_data.csv
git add data/daily_data.csv.dvc .gitignore
git commit -m "Add initial version of the dataset"
dvc remote add -d myremote gdrive://your_folder_id

dvc push
git tag -a v1.0 -m "Version 1.0 of the dataset"
git push origin v1.0


# Update or replace data/daily_data.csv
dvc add data
git add data.dvc
git commit -m "Update dataset to version 2.0"
dvc push
git tag -a v2.0 -m "Version 2.0 of the dataset"
git push origin v2.0


git checkout v1.0
dvc pull

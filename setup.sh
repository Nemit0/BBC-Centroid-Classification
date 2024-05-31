# Init setup
# This assumes linux environment

# Make directories
mkdir ./data ./models ./data/data_chunks ./embed_chunks

# Get dataset from https://www.kaggle.com/datasets/jjinho/wikipedia-20230701/data
curl https://storage.googleapis.com/kaggle-data-sets/3521629/6146260/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240531%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240531T033129Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=604999b486122db8055828c6f1b1c5bb7d378427682fb1ab71fc212f7a77f041d80a68fc57e1f4d07c1691a47580e2f2059dbfd8fbab0f39997135905291f4cf7f976d723b9aa4660d5aceb5dd9baa302fc81fda7b721b9964e252fb883cad3200a6dd304f808438246e0c61d3b1d2ccda9bcf85a307a7659c7eb47a673ff1d05047097ddf92e0585ad449fd22484cd6d8d6bf339608817042fd52693bf08395d8edd962ed18a6af938b76cee0ef69ac3f955b76ce76063518c47735409aac97c8cff7295fe246b362749ef24fdb6c1faf93865d368e9a6ec0ace83d63097fbea0e91b462cabe84ac7d308536fec6ec4ad71733caa416403df7dc7a20cf1a4bf -o ./dataset.zip

# Unzip dataset to ./data folder
unzip ./dataset.zip -d ./data

echo "Dataset fetched, unzipped and stored in ./data folder"
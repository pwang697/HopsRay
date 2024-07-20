#!/bin/bash

KUBE_NAMESPACE=$1
KUBE_SECRET=$2

CERTS_DIR=secrets
PEMS_DIR=secrets/pems

mkdir -p $CERTS_DIR
mkdir -p $PEMS_DIR

# Retrieve the Kubernetes secret and save it to a file
kubectl get secret "$KUBE_SECRET" -n "$KUBE_NAMESPACE" -o json > "$CERTS_DIR/secret.json"

# Verify that the file was created successfully
if [ ! -f "$CERTS_DIR/secret.json" ]; then
  echo "Failed to save the secret to file."
  exit 1
fi

# Extract the secret data using jq
secret_data=$(jq -r '.data' "$CERTS_DIR/secret.json")
if [ -z "$secret_data" ]; then
  echo "No secret data found."
  exit 2
fi

# Loop through each key and decode the value
for key in $(echo "$secret_data" | jq -r 'keys[]'); do
  value=$(echo "$secret_data" | jq -r --arg key "$key" '.[$key]')
  echo "$value" | base64 --decode > "$CERTS_DIR/$key"
  if [ $? -eq 0 ]; then
    echo "Decoded and saved $CERTS_DIR/$key"
  else
    echo "Failed to decode $CERTS_DIR/$key"
    exit 3
  fi
done

HADOOP_USERNAME=${KUBE_SECRET//-/_}
TSTORE_FILE=$CERTS_DIR/${HADOOP_USERNAME}__tstore.jks
KSTORE_FILE=$CERTS_DIR/${HADOOP_USERNAME}__kstore.jks
KEY_FILE=$CERTS_DIR/${HADOOP_USERNAME}__cert.key

KEY=$( cat ${KEY_FILE} )

#1. convert to certificates pem
keytool -importkeystore -srckeystore $KSTORE_FILE -destkeystore $PEMS_DIR/${HADOOP_USERNAME}__keystore.p12 -deststoretype PKCS12 -srcstorepass $KEY -keypass $KEY -deststorepass $KEY
openssl pkcs12 -nokeys -in $PEMS_DIR/${HADOOP_USERNAME}__keystore.p12 -out $PEMS_DIR/${HADOOP_USERNAME}_certificate_bundle.pem -passin pass:$KEY -legacy

#2. convert to root ca pem
keytool -importkeystore -srckeystore $TSTORE_FILE -destkeystore $PEMS_DIR/${HADOOP_USERNAME}__tstore.p12 -deststoretype PKCS12 -srcstorepass $KEY -keypass $KEY -deststorepass $KEY
openssl pkcs12 -nokeys -in $PEMS_DIR/${HADOOP_USERNAME}__tstore.p12 -out $PEMS_DIR/${HADOOP_USERNAME}_root_ca.pem -passin pass:$KEY -legacy

#3 convert to private key pem
openssl pkcs12 -info -in $PEMS_DIR/${HADOOP_USERNAME}__keystore.p12 -nodes -nocerts > $PEMS_DIR/${HADOOP_USERNAME}_private_key.pem -passin pass:$KEY -legacy

#4. verify that files have been created
CERTIFICATES_BUNDLE=$PEMS_DIR/${HADOOP_USERNAME}_certificate_bundle.pem
if [ ! -f ${CERTIFICATES_BUNDLE} ]; then
echo "Failed to convert keystore to certificate bundle pem format for $HADOOP_USERNAME"
exit 4
fi

ROOT_CA=$PEMS_DIR/${HADOOP_USERNAME}_root_ca.pem
if [ ! -f ${ROOT_CA} ]; then
echo "Failed to convert trust store to root ca pem format for $HADOOP_USERNAME"
exit 5
fi

PRIVATE_KEY=$PEMS_DIR/${HADOOP_USERNAME}_private_key.pem
if [ ! -f ${PRIVATE_KEY} ]; then
echo "Failed to covert .jks key to private key pem format  for $HADOOP_USERNAME"
exit 6
fi

chmod 640 $ROOT_CA
chmod 640 $CERTIFICATES_BUNDLE
chmod 640 $PRIVATE_KEY

rm -f $PEMS_DIR/${HADOOP_USERNAME}__keystore.p12
rm -f  $PEMS_DIR/${HADOOP_USERNAME}__tstore.p12

# Encode the new PEM files to base64
CERTIFICATES_BUNDLE_B64=$(base64 -w 0 $CERTIFICATES_BUNDLE)
ROOT_CA_B64=$(base64 -w 0 $ROOT_CA)
PRIVATE_KEY_B64=$(base64 -w 0 $PRIVATE_KEY)

# Create a patch file
cat <<EOF > $CERTS_DIR/patch.json
{
  "data": {
    "${HADOOP_USERNAME}_certificate_bundle.pem": "$CERTIFICATES_BUNDLE_B64",
    "${HADOOP_USERNAME}_root_ca.pem": "$ROOT_CA_B64",
    "${HADOOP_USERNAME}_private_key.pem": "$PRIVATE_KEY_B64"
  }
}
EOF

# Patch the secret with the new PEM files
kubectl patch secret "$KUBE_SECRET" -n "$KUBE_NAMESPACE" --patch "$(cat $CERTS_DIR/patch.json)"
echo "Secret $KUBE_SECRET updated with new PEM files."

rm -rf $PEMS_DIR
rm -rf $CERTS_DIR
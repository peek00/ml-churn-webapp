#!/bin/sh

# Run npm run dev in the background in detached mode

# Wait for the application to start (you can adjust the sleep duration as needed)
# sleep 5

ROOT_DIR=./dist
# Replace env vars in files served by NGINX
for file in $ROOT_DIR/js/*.js* $ROOT_DIR/index.html $ROOT_DIR/precache-manifest*.js;
do
  sed -i 's|VITE_VUE_APP_PREDICT_URL_PLACEHOLDER|'"fuck you"'|g' $file
  # sed -i 's|VITE_VUE_APP_PREDICT_URL_PLACEHOLDER|'${VITE_VUE_APP_PREDICT_URL}'|g' $file
  # Your other variables here...
done

npm run build
npm run dev -- --host 0.0.0.0

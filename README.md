# team08
xunyi and darion

AWS Links (Xun Yi) : http://ec2-13-229-61-253.ap-southeast-1.compute.amazonaws.com/
AWS Links (Darion) : http://3.27.92.215


--- 
Questions to ask:

- Do we maintain class imbalance when doing the train-test split, or only within train-validation split? Or do we use random split for train-test.

- 

To Do

- Fix preprocessing for checkboxes on frontend
- Implement plotly graphs
- 

---
# Takeaways

* Same min-max scaler object can be used for multiple columns
* One hot encoding requires different object for different columns


```
ssh -i .ssh/ai300-capstone.pem ec2-user@18.143.132.162
ssh -i .ssh/ai300-capstone.pem ec2-user@13.212.19.232 
ssh -i .ssh/ubuntu-ml-app.pem ec2-user@18.143.147.48 

docker run -d -p=80:80 --name=ml-app peek00/churn-prediction-ml-app
docker run -d -p=8000:80 --name=ml-app peek00/churn-prediction-ml-app:prod

DOMAIN NAME: https://peekoo-churn-ml-app-backend.crabdance.com/
```
cd /etc/pki/tls/certs
sudo openssl genpkey -algorithm RSA -out localhost.key
sudo openssl req -new -key localhost.key -out localhost.csr
sudo openssl x509 -req -days 365 -in localhost.csr -signkey localhost.key -out localhost.crt

[ec2-user@ip-172-31-32-74 certs]$ ls -a
.  ..  ca-bundle.crt  ca-bundle.trust.crt  localhost.crt  localhost.csr  localhost.key


sudo certbot certonly --nginx --agree-tos --email xunyi.work+ssl@gmail.com --no-eff-email -d peekoo-churn-ml-app-backend.crabdance.com

sudo nano /etc/nginx/sites-available/peekoo-churn-ml-app-backend.crabdance.com

server {
    listen 443 ssl;
    server_name peekoo-churn-ml-app-backend.crabdance.com;

    ssl_certificate /etc/letsencrypt/live/peekoo-churn-ml-app-backend.crabdance.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/peekoo-churn-ml-app-backend.crabdance.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;  # Replace with your Docker container's address
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

sudo ln -s /etc/nginx/sites-available/peekoo-churn-ml-app-backend.crabdance.com /etc/nginx/sites-enabled/

sudo systemctl restart nginx

---

To Do List
- Plot the AUC graph to determine best threshold
- Clean up code and comments

---
Front End readme

- Write hook that constantly polls the backend for the preprocessing, allows them to get scaled results instantly
- 

Things to PCA on
["contract_type", "tenure_months", "total_long_distance_fee", "total_charges_quarter", "has_premium_tech_support", "num_dependents" ]

docker build -t ml-churn-backend -f Dockerfile.backend .


---

ubuntu@ip-172-31-36-124:~$ sudo certbot certonly --nginx --agree-tos --email xunyi.work+ssl@gmail.com --no-eff-email -d peekoo-churn-ml-app-backend.crabdance.com
Saving debug log to /var/log/letsencrypt/letsencrypt.log
Account registered.
Requesting a certificate for peekoo-churn-ml-app-backend.crabdance.com

Successfully received certificate.
Certificate is saved at: /etc/letsencrypt/live/peekoo-churn-ml-app-backend.crabdance.com/fullchain.pem
Key is saved at:         /etc/letsencrypt/live/peekoo-churn-ml-app-backend.crabdance.com/privkey.pem
This certificate expires on 2023-10-03.
These files will be updated when the certificate renews.
Certbot has set up a scheduled task to automatically renew this certificate in the background.





# The MishapS of the SSL and the HTTPS

Description:

The issue arose when I deployed my front end on `Vercel` which hosted the website on a `https` connection and my backend on `AWS EC2` which hosted the website on a `http` connection. This caused the browser to block the request from the front end to the backend due to the `Mixed Content` issue.

--- 

The following is a summary of the steps I took to resolve the issue:

1. Ensure that when starting an Amazon EC2 instance to pick `Ubuntu` instead of `Amazon Linux`. The latter proved to have a lack of support for packages like `Certbot` which was very helpful in obtaining a **self signed** SSL certificate.
2. Created a domain using `https://freedns.afraid.org/` and pointed it to the public IP address of the EC2 instance. This allowed me to use the domain address created when I created the SSL certificate.
3. Installed `Certbot` and obtained a self signed SSL certificate using the following commands:

```
# Install certbot and nginx(?)
sudo apt update
sudo apt install certbot python3-certbot-nginx

# Generate self-signed certificate
sudo certbot certonly --nginx --agree-tos --email <replace this> --no-eff-email -d <peekoo-churn-ml-app-backend.crabdance.com>

# Open up nginx.conf (I think)
sudo nano /etc/nginx/sites-available/peekoo-churn-ml-app-backend.crabdance.com

# Pasted the following block into the file and saved it
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

# Created a symbolic link to the sites-enabled folder
sudo ln -s /etc/nginx/sites-available/peekoo-churn-ml-app-backend.crabdance.com /etc/nginx/sites-enabled/

# Restarted nginx
sudo systemctl restart nginx

```

After this was done, the instance should be able to be connected via https. It will throw a warning that it is a self signed certificate but requests can be made to it. The following steps are just general steps to serving a Docker file:

1. Install `docker` on the EC2 instance [here](https://docs.docker.com/engine/install/ubuntu/)
2. sudo docker run -d -p 8000:80 --name ml-app peek00/churn-prediction-ml-app:prod

---
### Ensure that for -p 8000:80, the first port matches the one defined in the block above and the second to the one defined in the Dockerfile.!
---

# Mistakes I made!

1. Took me too long to realise that instead of using the default Amazon Linux image when creating an EC2 instance and struggling with `sudo yum`, I could have just selected the `ubuntu` image when creatig the instance and following the more conventional and easier approach of `sudo apt` and the wider access to more repository, especially Certbot.

2. Regarding `env` variables and the interaction with `Vue` and `Vite`,
    1. `.env` values are statically embedded into the build at build time. This means that if you change the values in the `.env` file, you will need to rebuild the project to see the changes. This poses a problem when using `docker-compose up` as the `hostname` might not resolve to the correct IP address and becomes a string value.
    2. You have to prefix the environment variables with `VITE_` in order for them to be accessible in the `Vite` environment. This is because `Vite` is a build tool and not a runtime environment. This means that the environment variables are not accessible at runtime and must be prefixed with `VITE_` to be accessible at build time. I also prefixed it with `VUE` just in case, so my env variable looks like `VITE_VUE_APP_PREDICT_URL`. I was UNABLE to get it done the "proper" way in a `docker-compose up`.
    3. If using `Postman` to send request to a self signed SSL cert website, remember to check off the `SSL certificate verification` option in the settings. Otherwise it will not be able to send a query to the website.
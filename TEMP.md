# What I've Learnt

Through AI300, I've learnt the following:

* Converting machine learning notebook into a Object-Oriented Programming (OOP) style with Python scripts.
* Building abstract base classes in Python to make swapping out models, preprocessing methodologies and data sources easier.
* Pickling models and preprocessing objects to be used in production.
* Understanding the differences between a Flask Application and a production server.
* Dockerizing and deploying an image to AWS EC2.

Beyond the course goals, I tried in several ways to challenge myself and ran into unexpected issues that I had to troubleshoot on my own. We learnt to package a Flask application with a frontend built using static files in Flask but I decided to try and deploy a frontend build using Vue instead. I deployed my frontend using Vercel and my backend using Amazon EC2, but I ran into issues with SSL certificates and had to learn how to configure my backend to use HTTPS. 

This involved: 
* Getting a domain name using FreeDNS
* Generating a self-signed certificate using Certbot
* Setting up CORS 

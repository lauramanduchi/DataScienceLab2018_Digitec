# Infrastructure @ DG

- We installed a VM in the google cloud for you. 

## Machine Setup
- PostgreSQL DB with all the datasets
- Pipenv with a preconfigured virtualenvironment:
    - Installed packages:
        - tensorflow
        - sklearn
        - numpy
        - scipy
        - pandas
        - seaborn for plotting
        - sqlalchemy to access SQL Server
    - pipenv install ... to install a package
    - pipenv run python ... to run a python script
    - pipenv shell to get a shell with activated virtualenv
- Jupyter
    - cd /home/dslab/dslab && start_jupyter.sh to start the server
    - Quit in the Browser to stop it
- If you need any tools etc. just tell us we will set it up.

### GPU
- Per default there is no GPU attached to the machine
- If you need one just tell us and we will upgrade the machine

## Access the Machine
- Just send us your SSH public key we will add you to the authorized users.
- Your user is dslab everything you need is in the home directory
- IP Adress is 35.240.23.69
    - sudo echo "35.240.23.69 dslab-vm" >> /etc/hosts to reference it using dslab-vm on your local machine
- Use SSH Tunneling to access ports on the VM
    - ssh -L 8888:localhost:8888 dslab@dslab-vm to connect
    - Then you can access jupyter under localhost:8888 on your client machine

## Access to db
- Log in to the machine
- psql -d dslab dslab (Login to db dslab with user dslab)
    - PW: dslab2018
- \dt to list tables
- \d+ tablename to view schema
- Execute queries directly here
- Access to db in code using sqlalchemy:
    - engine = sqlalchemy.create_engine('postgresql://dslab:dslab2018@localhost/dslab')
- The database itself is owned by dslab, therefore if you want to create indices, views whatever feel free.

## Preloaded Data
- You basically have three tables:
    - traffic
    - product_purchase
    - product
- In case the db breaks there is a jupyter notebook ready to import all the data again into the notebook.

### traffic
- One request to our Backend Webservers per row

### product
- For each Product and Property Combination we have one row.
- A PropertyValueId only makes sense in combination with a PropertyTypeId.

### product_purchase
- Each row is a purchased product. If multiples of the same product is purchased in one order it is still one row.

### Known Problems
- To map the selected PropertyValue and PropertyType from the URL you need a function. We will provide that.
- We had a error in the Tracking component, and therefore the prices for so called 'ProductSets' are wrong. I have excluded the 'ProductSets' from the DB dumps, so if you have a product where you cannot find any data in the product table you can just ignore it.
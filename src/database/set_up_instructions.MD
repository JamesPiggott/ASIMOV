wget https://ftp.postgresql.org/pub/source/v15.2/postgresql-15.2.tar.gz
sudo tar xzf postgresql-15.2.tar.gz
sudo nano postgresql-15.2/contrib/cube/cubedata.h

# Change CUBE_MAX_DIM  from 100 to requisite number. In our case 256
# Save with Ctrl-X and then confirm with Y

cd postgresql-15.2  # Just make sure you have navigated into the untarred folder

sudo apt install libreadline-dev # This library is required by the configure script, so just perform the install to be sure

./configure
make
sudo -s
make install
mkdir /usr/local/pgsql/data
chown [USER] /usr/local/pgsql/data
/usr/local/pgsql/bin/initdb -D /usr/local/pgsql/data
/usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data -l
/usr/local/pgsql/bin/createdb test
/usr/local/pgsql/bin/psql test

cd contrib/cube   # From postgresql-15.2, the previous commands should not change directory

make
sudo -s
make install

# Manually create a database
mkdir /usr/local/pgsql/data
chown kane /usr/local/pgsql/data
/usr/local/pgsql/bin/initdb -D /usr/local/pgsql/data
/usr/local/pgsql/bin/pg_ctl -D /usr/local/pgsql/data -l
/usr/local/pgsql/bin/createdb face_vector_db
/usr/local/pgsql/bin/psql face_vector_db
CREATE USER face WITH PASSWORD 'face';
GRANT ALL PRIVILEGES ON DATABASE face_vector_db TO face;
CREATE EXTENSION cube;

CREATE TABLE face_encodings
(
    encodeId serial,
    name varchar,
	email varchar,
    face_vector cube
);

CREATE INDEX face_vector_idx ON face_encodings(face_vector);
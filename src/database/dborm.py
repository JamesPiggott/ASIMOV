import psycopg2
import psycopg2.extras
from src import Settings

class DBORM(object):
    """
    DBORM is the lowest level PostgreSQL class in the application. All other database interactions make use of the
    functions described below.
    """

    def __init__(self):
        psycopg2.extras.register_uuid()
        self.conn, self.cur = self.connect_to_database()

    def connect_to_database(self):
        """
        Connect to the database using psycopg2
        """
        try:
            conn = psycopg2.connect("dbname=" + Settings.database_name + " " +
                                    "user=" + Settings.user_name + " " +
                                    "password=" + Settings.password)
            return conn, conn.cursor()
        except (Exception, psycopg2.DatabaseError) as error:
            print("connect_to_database " + str(error))

    def execute_sql_command(self, sql_commands):
        try:
            self.cur.execute(sql_commands)
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("execute_sql_command " + str(error))

    def execute_update_sql_command(self, table_name, column_names, column_name, filter_argument):
        """
        Update a current record in the database
        """
        try:
            self.cur.execute("UPDATE " + table_name + " SET " + column_names + " WHERE " + column_name + "='"
                             + filter_argument + "';")
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("execute_update_sql_command " + str(error))

    def execute_insert_sql_command(self, table_name, column_names, column_place_holder, data_values):
        """
        Insert a new record into the database
        """
        try:
            self.cur.execute("INSERT INTO " + table_name + "(" + column_names + ") VALUES (" + column_place_holder + ")"
                             , data_values)
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("execute_insert_sql_command " + str(error))

    def execute_delete_sql_command(self, table_name, column_name, filter_argument):
        """
        Delete a record from the database
        """
        try:
            self.cur.execute("DELETE FROM " + table_name + " WHERE " + column_name + "='" + filter_argument + "';")
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("execute_delete_sql_command " + str(error))

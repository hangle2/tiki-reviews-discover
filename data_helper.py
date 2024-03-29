import json
import logging

import mysql.connector
from mysql.connector import errorcode


# return a list of reviews
def get_reviews_from_database(product_id=4536405):
    reviews = []
    try:
        with open('database-config.json') as f:
            config = json.load(f)
            logging.info('using database config %s', config)
        connection = mysql.connector.connect(**config)

        sql_select_query = "SELECT title, content FROM talaria_review.review where product_id=" + str(product_id)
        cursor = connection.cursor()
        cursor.execute(sql_select_query)
        records = cursor.fetchall()
        for x in records:
            reviews.append({'title': x[0], 'content': x[1]})
        logging.info("Fetched %d reviews for product_id=%d", cursor.rowcount, product_id)

        connection.close()
        cursor.close()
        logging.info("MySQL connection is closed")
        return reviews
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        exit(-1)

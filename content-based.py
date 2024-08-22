
import mysql.connector
from mysql.connector import Error
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Thay đổi mã hóa mặc định của sys.stdout sang utf-8
sys.stdout.reconfigure(encoding='utf-8')

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

# Thông tin kết nối
host_name = ""
user_name = ""
user_password = "123456"
db_name = ""

# Tạo kết nối
connection = create_connection(host_name, user_name, user_password, db_name)

def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result, cursor.column_names
    except Error as e:
        print(f"The error '{e}' occurred")

# Query để lấy thông tin bác sĩ
select_users = """
SELECT doctor_infors.*, users.image, users.firstName, users.lastName, description_alls.description, role_codes.valueVI AS roleValue, position_codes.valueVI AS positionValue, specialties.name AS nameSpecialty, health_facilities.name AS nameHealthFacilities, GROUP_CONCAT(DISTINCT DATE(FROM_UNIXTIME(schedules.date / 1000)) ORDER BY FROM_UNIXTIME(schedules.date / 1000) SEPARATOR ' ') AS dates, 
       GROUP_CONCAT(DISTINCT schedules.timeType ORDER BY schedules.timeType SEPARATOR ' ') AS timeTypes
FROM doctor_infors
JOIN users ON doctor_infors.doctorId = users.id
JOIN description_alls ON doctor_infors.doctorId = description_alls.doctorId
JOIN allcodes AS role_codes ON users.roleId = role_codes.keyMap
JOIN allcodes AS position_codes ON users.positionId = position_codes.keyMap
JOIN schedules ON doctor_infors.doctorId = schedules.doctorId
JOIN specialties ON doctor_infors.specialtyId = specialties.id
JOIN health_facilities ON doctor_infors.healthFacilitiesId = health_facilities.id
GROUP BY doctor_infors.doctorId
"""

users, columns = execute_read_query(connection, select_users)

# Chuyển đổi users thành DataFrame
users_df = pd.DataFrame(users, columns=columns)
print(users_df)
# Thêm cột combineFeatures
def combine_features(row):
    return f"{row['provinceId']},  {row['priceId']}, {row['nameSpecialty']},  {row['healthFacilitiesId']}"
    #  return f"{row['dates']}, {row['timeTypes']}"

users_df['combine_features'] = users_df.apply(combine_features, axis=1)
print(users_df['combine_features'])
# Tạo DataFrame mới chỉ chứa cột combineFeatures
combine_features_df = users_df[['combine_features']]


def custom_tokenizer(text):
    return text.split(', ')

# # # smooth_idf=True, norm ='l2'
tf = TfidfVectorizer(tokenizer=custom_tokenizer)


# Tạo TF-IDF vectorizer
# tf = TfidfVectorizer()
tfMatrix = tf.fit_transform(users_df['combine_features'])

print(tfMatrix)

# # # Get the vocabulary (terms) learned by TfidfVectorizer
vocabulary = tf.vocabulary_

# # Get the IDF (Inverse Document Frequency) learned by TfidfVectorizer
idf_values = tf.idf_

# # Print vocabulary and IDF values
print("Vocabulary (Terms):")
print(vocabulary)
print("\nIDF Values:")
print(idf_values)

# # Print TF-IDF matrix
# print("\nTF-IDF Matrix:")
# print(tfidf_matrix.toarray())

# # Print feature names (terms)
feature_names = tf.get_feature_names_out()
print("\nFeature Names (Terms):")
print(feature_names)

# Tính toán độ tương đồng cosine
# cosine_sim = cosine_similarity(tfMatrix)
# print(cosine_sim)



@app.route('/api/recommend-doctors', methods=['GET'])
def recommend_doctors():
    patient_id = request.args.get('id')
    if patient_id is None:
        return jsonify({'error': 'Patient ID is missing'}), 400

    try:
        patient_id = int(patient_id)
    except ValueError:
        return jsonify({'error': 'Invalid Patient ID'}), 400

    # Truy vấn lịch sử khám bệnh
    query_history = f"""
    SELECT patientId, doctorId, reasonExamination
    FROM bookings
    WHERE patientId = {patient_id} AND statusId = 'S3'
    """
    history, history_columns = execute_read_query(connection, query_history)
    history_df = pd.DataFrame(history, columns=history_columns)
    print(history_df)
    if history_df.empty:
        return jsonify({'error': 'No history found for this patient'}), 400

    # Xây dựng hồ sơ bệnh nhân dựa trên các bác sĩ đã khám
    patient_doctors = history_df['doctorId'].values
    # patient_doctors = list(set(history_df['doctorId'].values))
    print(patient_doctors)
    # patient_profile = np.asarray(tfMatrix[users_df[users_df['doctorId'].isin(patient_doctors)].index].mean(axis=0))

    # print(patient_profile)

    # # Tính toán độ tương đồng cosine giữa hồ sơ bệnh nhân và hồ sơ các bác sĩ
    # similarity_scores = cosine_similarity(patient_profile, tfMatrix)
    # print(similarity_scores)
    # # Sắp xếp các bác sĩ theo độ tương đồng
    # sorted_indices = similarity_scores.argsort()[0][::-1]

    # top_doctors = sorted_indices[:5]
    # print(top_doctors)

     # Tính tần suất khám của các bác sĩ
    doctor_frequencies = history_df['doctorId'].value_counts().to_dict()
    # print(doctor_frequencies.values())
    # print('sum: ',sum(doctor_frequencies.values()))

    # Hồ sơ bệnh nhân dựa trên tần suất khám
    patient_profile = np.zeros(tfMatrix.shape[1])
    print(patient_profile)
    for doctor_id, freq in doctor_frequencies.items():
        doctor_index = users_df[users_df['doctorId'] == doctor_id].index[0]
        patient_profile += tfMatrix[doctor_index].toarray()[0] * freq
        # print(patient_profile)
    patient_profile /= sum(doctor_frequencies.values())
    print('check patient profile: ', patient_profile)
    # print('check patient profile: ', patient_profile.reshape(1, -1))

    # Tính toán độ tương đồng cosine giữa hồ sơ bệnh nhân và hồ sơ các bác sĩ
    similarity_scores = cosine_similarity(patient_profile.reshape(1, -1), tfMatrix)
    print('check cosine: ', similarity_scores)
    
    # Sắp xếp các bác sĩ theo độ tương đồng
    sorted_similarity = sorted(enumerate(similarity_scores[0]), key=lambda x: x[1], reverse=True)
    print(sorted_similarity)

    # lấy 6 doctor có độ tương đồng cao nhất
    top_doctors = [index for index, score in sorted_similarity[:6]]
    print(top_doctors)


    result = []
    for index in top_doctors:
        row = users_df.iloc[index]
        result.append({
            'description': row['description'],
            'firstName': row['firstName'],
            'lastName': row['lastName'],
            'roleValue': row['roleValue'],
            'positionValue': row['positionValue'],
            'image': row['image'].decode('utf-8') if isinstance(row['image'], bytes) else row['image'],
            'doctorId': int(row['doctorId']),
            'nameSpecialty': row['nameSpecialty']
        })

    return jsonify({'recommended_doctors': result})


if __name__ == '__main__':
    app.run(port=6969)

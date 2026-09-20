import os
import pandas as pd
import yaml

def load_config():
    with open("configs/data_config.yaml", "r") as f:
        return yaml.safe_load(f)

def validate_raw_data():
    config = load_config()
    raw_dir = config["raw_data_dir"]
    expected_files = config["expected_files"]
    
    for f in expected_files:
        path = os.path.join(raw_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    # Check studentInfo
    info_path = os.path.join(raw_dir, "studentInfo.csv")
    df_info = pd.read_csv(info_path)
    
    # Check composite key uniqueness
    dups = df_info.duplicated(subset=['id_student', 'code_module', 'code_presentation'])
    if dups.any():
        raise ValueError("Duplicate composite key in studentInfo")
        
    valid_targets = {"Pass", "Withdrawn", "Fail", "Distinction"}
    if not df_info['final_result'].isin(valid_targets).all():
        raise ValueError("Invalid target value in studentInfo")
        
    info_keys = set(zip(df_info['id_student'], df_info['code_module'], df_info['code_presentation']))
    
    # Check courses
    courses_path = os.path.join(raw_dir, "courses.csv")
    df_courses = pd.read_csv(courses_path)
    course_keys = set(zip(df_courses['code_module'], df_courses['code_presentation']))

    # Check studentRegistration
    reg_path = os.path.join(raw_dir, "studentRegistration.csv")
    df_reg = pd.read_csv(reg_path)
    reg_keys = set(zip(df_reg['id_student'], df_reg['code_module'], df_reg['code_presentation']))
    if not reg_keys.issubset(info_keys):
        raise ValueError("studentRegistration key not in studentInfo")
    
    reg_course_keys = set(zip(df_reg['code_module'], df_reg['code_presentation']))
    if not reg_course_keys.issubset(course_keys):
        raise ValueError("studentRegistration course not in courses")

    # Check assessments
    assessments_path = os.path.join(raw_dir, "assessments.csv")
    df_asm = pd.read_csv(assessments_path)
    assessment_keys = set(df_asm['id_assessment'])
    
    # Check studentAssessment
    student_assessment_path = os.path.join(raw_dir, "studentAssessment.csv")
    df_s_asm = pd.read_csv(student_assessment_path)
    if not set(df_s_asm['id_assessment']).issubset(assessment_keys):
        raise ValueError("studentAssessment id_assessment not in assessments")

    # Check vle
    vle_path = os.path.join(raw_dir, "vle.csv")
    df_vle = pd.read_csv(vle_path)
    vle_keys = set(zip(df_vle['id_site'], df_vle['code_module'], df_vle['code_presentation']))

    # Check studentVle
    student_vle_path = os.path.join(raw_dir, "studentVle.csv")
    vle_total_rows = 0
    chunksize = 1000000
    for chunk in pd.read_csv(student_vle_path, chunksize=chunksize):
        chunk_keys = set(zip(chunk['id_site'], chunk['code_module'], chunk['code_presentation']))
        if not chunk_keys.issubset(vle_keys):
            raise ValueError("studentVle site not in vle")
        vle_total_rows += len(chunk)
            
    print(f"Raw data validation passed. Total studentVle rows validated: {vle_total_rows}")

if __name__ == "__main__":
    validate_raw_data()

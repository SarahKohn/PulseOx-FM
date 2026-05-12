# Setup
import warnings
import sys
import os
import random
import torch
import math
import numpy as np
import pandas as pd
import wandb
from typing import Iterable

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(PARENT_DIR)
SEED = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_FOR_CSV = '/net/mraid20/export/jafar/Sarah/csv_files/'
PROJECT_DIR = '/net/mraid20/export/jafar/SleepFM/ssl_sleep/'
PERSONAL_DIR = '/net/mraid20/export/jafar/Sarah/ssl_sleep/'  # change to your personal folder or the shared one (PROJECT_DIR)
# for running on old dataset, recordings indexed by date
RECORDS_PATH = os.path.join(PROJECT_DIR, 'tabular_data', 'recordings_with_labels2.csv')
ITAMAR_PATH = os.path.join(PROJECT_DIR, 'tabular_data', 'recordings_with_itamar_features.csv')
PYPPG_PATH = os.path.join(PROJECT_DIR, 'tabular_data', 'full_sleep_data_with_recording_name.csv')
# gold dataset: recordings indexed by research_stage
GOLD_ITAMAR_PATH = os.path.join(PROJECT_DIR, 'tabular_data', 'itamar_data_gold.csv')
GOLD_PYPPG_PATH = os.path.join(PROJECT_DIR, 'tabular_data', 'pyppg_data_gold.csv')
GOLD_RECORDS_PATH = os.path.join(PROJECT_DIR, 'tabular_data', 'recordings_with_labels_gold.csv')
NORMALIZED_LABELS_PATH = os.path.join(PROJECT_DIR, 'tabular_data', 'train_recordings_with_normalized_labels_for_multilosss.csv')

# Shared grouping for baseline diagnoses / incidence analyses.
DISEASE_CATEGORY_MAP = {
    # Cardiovascular disorders
    "Hypertension": "Cardiovascular disorders",
    "Ischemic Heart Disease": "Cardiovascular disorders",
    "Heart valve disease": "Cardiovascular disorders",
    # Metabolic, endocrine and reproductive disorders
    "Obesity": "Metabolic, endocrine and reproductive disorders",
    "Hyperlipidemia": "Metabolic, endocrine and reproductive disorders",
    "Prediabetes": "Metabolic, endocrine and reproductive disorders",
    "Gout": "Metabolic, endocrine and reproductive disorders",
    "Fatty Liver Disease": "Metabolic, endocrine and reproductive disorders",  # MASLD
    "Hashimoto": "Metabolic, endocrine and reproductive disorders",
    "Polycystic Ovary Disease": "Metabolic, endocrine and reproductive disorders",
    "Endometriosis and Adenomyosis": "Metabolic, endocrine and reproductive disorders",
    # # Metabolic conditions
    # "Obesity": "Metabolic conditions",
    # "Hyperlipidemia": "Metabolic conditions",
    # "Prediabetes": "Metabolic conditions",
    # "Gout": "Metabolic conditions",
    # "Fatty Liver Disease": "Gastrointestinal disorders",  # MASLD
    # # Endocrine / Reproductive disorders
    # "Hashimoto": "Endocrine and Reproductive disorders",
    # "Polycystic Ovary Disease": "Endocrine and Reproductive disorders",
    # "Endometriosis and Adenomyosis": "Endocrine and Reproductive disorders",
    # Gastrointestinal disorders
    "Gallstone Disease": "Gastrointestinal disorders",
    "IBS": "Gastrointestinal disorders",  # Irritable Bowel Syndrome
    "Peptic Ulcer Disease": "Gastrointestinal disorders",
    "Anal Fissure": "Gastrointestinal disorders",
    "Anal abscess": "Gastrointestinal disorders",
    "Haemorrhoids": "Gastrointestinal disorders",
    # Hematologic and Nutritional disorders
    "G6PD": "Hematologic and nutritional disorders",
    "B12 Deficiency": "Hematologic and nutritional disorders",
    "Anemia": "Hematologic and nutritional disorders",
    # Sleep disorders and mental health
    "Sleep Apnea": "Sleep disorders and mental health",
    "Insomnia": "Sleep disorders and mental health",
    "Anxiety": "Sleep disorders and mental health",
    "Depression": "Sleep disorders and mental health",
    # # Sleep disorders
    # "Sleep Apnea": "Sleep disorders",
    # "Insomnia": "Sleep disorders",
    # # Mental health
    # "Anxiety": "Mental health",
    # "Depression": "Mental health",
    # Neurological & Pain conditions
    "ADHD": "Neurological and pain conditions",
    "Migraine": "Neurological and pain conditions",
    "Headache": "Neurological and pain conditions",
    "Fibromyalgia": "Neurological and pain conditions",
    "Back Pain": "Neurological and pain conditions",
    # Immune / Allergic diseases
    "Allergy": "Immune and allergic diseases",
    "Asthma": "Immune and allergic diseases",
    "Psoriasis": "Immune and allergic diseases",
    "Atopic Dermatitis": "Immune and allergic diseases",
    "Vitiligo": "Immune and allergic diseases",
    # Other
    "Hearing loss": "Other",
    "Fractures": "Other",
    "Glaucoma": "Other",
    "Oral apthae": "Other",
    "Retinal detachment": "Other",
    "Sinusitis": "Other",
    "Urinary Tract Stones": "Other",
    "Urinary tract infection": "Other",
    "Osteoarthritis": "Other",
}

SIGNALS = [
    # # from Pulse oximetry sensor:
    'spo2',  # pulse oximetry sensor
    'heart_rate', # pulse oximetry sensor
    # # from PAT sensor:
    # 'heart_rate_raw', #PAT sensor??
    'pat_infra',  # PAT sensor used by Itamar for CSR events and for HRV calculation
    # 'pat_amplitude',  # PAT sensor for resp. events detection
    # 'pat_lpf',
    # 'pat_view',
    # # from wrist actigraphy:
    'actigraph', # (for sleep stages)
    # 'sleep_stage',
    # # from chest RESBP sensor:
    'respiratory_movement',  # chest actigraphy with mean removal
    'body_position',  # from local means of chest actigraphy
    'snore_db'  # chest microphone
]
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdirifnotexists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def load_dataset_filenames_dict() -> dict:
    return {'Age_Gender_BMI': os.path.join(PATH_FOR_CSV, 'age_gender_bmi_df.csv'),
            'Age_Gender_BMI_VAT': os.path.join(PATH_FOR_CSV, 'age_gender_bmi_vat_df.csv'),
            'nightingale': os.path.join(PATH_FOR_CSV, 'nightingale.csv'),
            'clinical_OSA': os.path.join(PATH_FOR_CSV, 'clinical_OSA.csv'),
            'pheno_sleep': os.path.join(PATH_FOR_CSV, 'all_nights_all_features_df.csv'),
            'pheno_sleep_avg': os.path.join(PATH_FOR_CSV, 'all_means_all_features_df.csv'),
            'sleep_key_features': os.path.join(PATH_FOR_CSV, 'all_nights_key_features_df.csv'),
            'sleep_key_features_v2': os.path.join(PATH_FOR_CSV, 'all_nights_key_features_df2.csv'), # from june-2024 with saturation_below_90
            'sleep_key_features_avg': os.path.join(PATH_FOR_CSV, 'all_means_key_features_df.csv'),
            'sleep_quality': os.path.join(PATH_FOR_CSV, 'all_nights_sleep_quality_features_df.csv'),
            'sleep_quality_avg': os.path.join(PATH_FOR_CSV, 'all_means_sleep_quality_features_df.csv'),
            'sleep_quality_filtered_avg': os.path.join(PATH_FOR_CSV, 'all_pheno_means_sleep_quality_df_filtered.csv'),
            'sleep_quality_best_night': os.path.join(PATH_FOR_CSV, 'all_pheno_best_sleep_quality_df.csv'),
            'hrv': os.path.join(PATH_FOR_CSV, 'all_nights_hrv_features_df.csv'),
            'hrv_avg': os.path.join(PATH_FOR_CSV, 'all_means_hrv_features_df.csv'),
            'blood_lipids': os.path.join(PATH_FOR_CSV, 'blood_lipids.csv'),
            'body_composition': os.path.join(PATH_FOR_CSV, 'body_composition.csv'),
            'bone_density': os.path.join(PATH_FOR_CSV, 'bone_density.csv'),
            'cardiovascular': os.path.join(PATH_FOR_CSV, 'cardiovascular_wo_distances.csv'),  # cardiovascular_system.csv
            'frailty': os.path.join(PATH_FOR_CSV, 'frailty.csv'),
            'glycemic_status': os.path.join(PATH_FOR_CSV, 'glycemic_status.csv'),
            'hematopoietic': os.path.join(PATH_FOR_CSV, 'hematopoietic_system.csv'),
            'immune_system': os.path.join(PATH_FOR_CSV, 'immune_system.csv'),
            'lifestyle': os.path.join(PATH_FOR_CSV, 'lifestyle.csv'),
            'lifestyle_numerical': os.path.join(PATH_FOR_CSV, 'lifestyle_numerical.csv'),
            'liver': os.path.join(PATH_FOR_CSV, 'liver.csv'),
            'renal_function': os.path.join(PATH_FOR_CSV, 'renal_function.csv'),
            'diet': os.path.join(PATH_FOR_CSV, 'diet.csv'),
            'diet_questions': os.path.join(PATH_FOR_CSV, 'diet_questions.csv'),
            'MB': os.path.join(PATH_FOR_CSV, 'microbiome.csv'),
            'medications': os.path.join(PATH_FOR_CSV, 'medications.csv'),
            'BM': os.path.join(PATH_FOR_CSV, 'body_measures_df.csv'),
            'MBspecies_not_log': os.path.join(PATH_FOR_CSV, 'species_counts_df_not_log.csv'),
            'MBspecies2_not_log': os.path.join(PATH_FOR_CSV, 'mb_species_counts_df_not_log.csv'),
            'MBspecies': os.path.join(PATH_FOR_CSV, 'species_counts_df.csv'),
            'MBgenus': os.path.join(PATH_FOR_CSV, 'genus_counts_df.csv'),
            'MBfamily': os.path.join(PATH_FOR_CSV, 'family_counts_df.csv'),
            'MBspecies2': os.path.join(PATH_FOR_CSV, 'all_mb_species_counts_df.csv'),
            'MBgenus2': os.path.join(PATH_FOR_CSV, 'all_mb_genus_counts_df.csv'),
            'MBfamily2': os.path.join(PATH_FOR_CSV, 'all_mb_family_counts_df.csv'),
            'MBorder2': os.path.join(PATH_FOR_CSV, 'all_mb_order_counts_df.csv'),
            'MBclass2': os.path.join(PATH_FOR_CSV, 'all_mb_class_counts_df.csv'),
            'MBphylum2': os.path.join(PATH_FOR_CSV, 'all_mb_phylum_counts_df.csv'),
            'MBpathways': os.path.join(PATH_FOR_CSV, 'all_mb_pathways_df.csv'),
            'retina': os.path.join(PATH_FOR_CSV, 'all_retina_df.csv'),
            'mental': os.path.join(PATH_FOR_CSV, 'all_mental_df.csv'),
            'baseline_diagnoses': os.path.join(PATH_FOR_CSV, 'baseline_diagnoses_df.csv'),
            'baseline_diagnoses_nastya': os.path.join(PATH_FOR_CSV, 'baseline_diagnoses_df_nastya.csv'),
            'baseline_diagnoses_nastya_raw':
                '/net/mraid20/export/genie/LabData/Data/10K/for_review/baseline_conditions_all.csv',
            'baseline_sleep_quality_avg': os.path.join(PATH_FOR_CSV, 'baseline_means_sleep_quality_features_df.csv'),
            'baseline_hrv_avg': os.path.join(PATH_FOR_CSV, 'baseline_means_hrv_features_df.csv'),
            'all_body_systems - Nastya_v2-wo_lipids': os.path.join(PATH_FOR_CSV,
                                                                   'all_body_systems_df-without_lipids.csv'),
            'all_body_systems': os.path.join(PATH_FOR_CSV, 'all_body_systems_df.csv')}


def load_dataset_to_name() -> dict:
    return {'Age_Gender_BMI': 'Age & BMI',
            'Age_Gender_BMI_VAT': 'Age, BMI & VAT',
            'hematopoietic': 'Hematopoietic system',
            'immune_system': 'Immune system',
            'glycemic_status': 'Insulin resistance',
            'lifestyle': 'Lifestyle',
            'lifestyle_numerical': 'Lifestyle',
            'mental': 'Mental health',
            'frailty': 'Frailty',
            'liver': 'Liver health',
            'renal_function': 'Renal function',
            'cardiovascular': 'Cardiovascular system',
            'body_composition': 'Body Composition',
            'bone_density': 'Bone Density',
            'blood_lipids': 'Blood lipids',
            'medications': 'Medications',
            'nightingale': 'NMR metabolomics',
            'MBspecies2': f'Gut MB species',
            'MBgenus2': f'Gut MB genus',
            'MBfamily2': f'Gut MB families',
            'MBphylum2': f'Gut MB phylum',
            'MBclass2': f'Gut MB class',
            'MBorder2': f'Gut MB order',
            'MBpathways': f'Gut MB metabolic pathways',
            'sleep': 'Sleep characteristics',
            'pheno_sleep': f'Sleep characteristics per night\n(with HRV)',
            'pheno_sleep_avg': f'Sleep characteristics\n(average of nights)',
            'sleep_quality_avg': f'Sleep Quality\n(average of nights)',
            'sleep_quality_filtered_avg': f'Sleep Quality\n(average of stable nights)',
            'sleep_quality_best_night': f'Sleep Quality\n(longest night)',
            'hrv_avg': f'HRV\n(average of nights)',
            'diet': 'Diet',
            'all_body_systems': f'All Body Systems\n(except sleep and Genetics)',
            'baseline_diagnoses': 'Diseases and medical conditions\nat baseline',
            'baseline_sleep_quality_avg': f'Sleep Quality at baseline\n',
            'baseline_hrv_avg': f'HRV at baseline\n'
            }


def features_renaming_dict() -> dict:
    return {
        'high_exercise_duration_Between an hour and an hour and a half':
            'high exercise (>1h)',
        'high_exercise_duration_Betweenanhourandanhourandahalf':
            'high exercise (>1h)',
        'falling_asleep_during_daytime_From time to time': 'falling asleep during daytime',
        'falling_asleep_during_daytime_Fromtimetotime': 'falling asleep during daytime',
        'nap_during_day_Fromtimetotime': 'nap during day - sometimes',
        'nap_during_day_Notorrarely': 'nap during day - no',
        'consider_yourself_morning_evening_Imtotallyamorningman': 'morning type person',
        'troubles_falling_a_sleep_Notorrarely': 'troubles falling asleep - no',
        'troubles_falling_a_sleep_Fromtimetotime': 'troubles falling asleep - sometimes',
        'tobacco_past_how_often_I smoked most or all days': 'smoked tobacco most days',
        'Progestogensandestrogenssystemiccontraceptivessequentialpreparations': 'Oral contraceptives',
        'ProtonpumpinhibitorsforpepticulcerandGORD': 'Proton pump inhibitors',
        'Preparationsinhibitinguricacidproduction': 'Uric acid production inhibitors',
        'Dihydropyridinederivativeselectivecalciumchannelblockerswithmainlyvasculareffects':
            'Dihydropyridines calcium channel blockers',
        'AngiotensinIIreceptorblockersARBsplain': 'Angiotensin receptor blockers',
        'Plateletaggregationinhibitorsexclheparin': 'Anti platlets',
        'Betablockingagentsselective': 'Beta blockers',
        'AngiotensinIIreceptorblockersARBsandcalciumchannelblockers':
            'Combination drug - ARBs & calcium channel blockers',
        'ACEinhibitorsplain': 'ACE Inhibitors',
        'HMGCoAreductaseinhibitorsplainlipidmodifyingdrugs': 'Statins',
        'bmi': 'BMI',
        'rds score': 'RDS score',
        'gender': 'sex',  # 'male'
        'ahi': 'pAHI',
        'odi': 'pODI',
        'hrv': 'PRV',
        'bmi': 'BMI',
        'number of ': '',
        'bt_': 'BT',
    }

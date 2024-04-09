class_set = ['Country', 'CountryCode', 'HardSep', 'Municipality', 'Name', 'OOV', 'PostalCode', 'Province', 'StreetName', 'StreetNumber', 'Unit']

locales = ['cs_CZ', 'da_DK', 'de_AT', 'de_CH', 'de_DE', 'en_GB', 'en_IE', 'en_IN', 'en_NZ', 'en_TH', 'en_US', 'es_CA', 'es_ES', 'es_MX', 'fr_CA', 'fr_CH', 'fr_FR', 'ga_IE', 'it_IT', 'nl_BE', 'nl_NL', 'no_NO', 'pl_PL', 'pt_BR', 'pt_PT', 'ro_RO', 'sl_SI', 'sv_SE', 'tr_TR']

prob_dict = {
    "Country": 0.8,
    "WhitespaceCut": 0.05,
    "Company": 0.7,
    "HardSep": 0.3
}

TAG2ID = {'B-Country': 0,
 'B-CountryCode': 1,
 'B-HardSep': 2,
 'B-Municipality': 3,
 'B-Name': 4,
 'B-OOV': 5,
 'B-PostalCode': 6,
 'B-Province': 7,
 'B-StreetName': 8,
 'B-StreetNumber': 9,
 'B-Unit': 10,
 'I-Country': 11,
 'I-Municipality': 12,
 'I-Name': 13,
 'I-OOV': 14,
 'I-PostalCode': 15,
 'I-Province': 16,
 'I-StreetName': 17,
 'I-StreetNumber': 18,
 'I-Unit': 19}
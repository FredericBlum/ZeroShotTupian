from helper_functions import conllu_split, concat_glove


################################
### data and dictionaries    ###
################################
akuntsu = conllu_split('../UD/UD_Akuntsu-TuDeT/aqz_tudet-ud-test.conllu', lang = 'Akuntsu')
guajajara = conllu_split('../UD/UD_Guajajara-TuDeT/gub_tudet-ud-test.conllu', lang = 'Guajajara')
kaapor = conllu_split('../UD/UD_Kaapor-TuDeT/urb_tudet-ud-test.conllu', lang = 'Kaapor')
karo = conllu_split('../UD/UD_Karo-TuDeT/arr_tudet-ud-test.conllu', lang = 'Karo')
makurap = conllu_split('../UD/UD_Makurap-TuDeT/mpu_tudet-ud-test.conllu', lang = 'Makurap')
munduruku = conllu_split('../UD/UD_Munduruku-TuDeT/myu_tudet-ud-test.conllu', lang = 'Munduruku')
tupinamba = conllu_split('../UD/UD_Tupinamba-TuDeT/tpn_tudet-ud-test.conllu', lang = 'Tupinamba')

# concat_glove(["Akuntsu", "Guajajara", "Kaapor", "Karo", "Makurap", "Munduruku", "Tupinamba"])
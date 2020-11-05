import pandas as pd
from faker import Faker # we use the package Faker to generate names

#Correspondance between language codes and names. (Taken from googletrans package, added dk for danish,tw for twi (ghana dialect))
LANGUAGES = {'af': 'afrikaans','am': 'amharic','ar': 'arabic','az': 'azerbaijani','be': 'belarusian','bg': 'bulgarian','bn': 'bengali',
             'bs': 'bosnian','ca': 'catalan','ceb': 'cebuano','co': 'corsican','cs': 'czech','cy': 'welsh','dk':'danish','da': 'danish','de': 'german',
             'el': 'greek','en': 'english','eo': 'esperanto','es': 'spanish','et': 'estonian','eu': 'basque','fa': 'persian','fi': 'finnish',
             'fr': 'french','fy': 'frisian','ga': 'irish','gd': 'scots gaelic','gl': 'galician','gu': 'gujarati','ha': 'hausa','haw': 'hawaiian',
             'he': 'hebrew','hi': 'hindi','hmn': 'hmong','hr': 'croatian','ht': 'haitian creole','hu': 'hungarian','hy': 'armenian','id': 'indonesian',
             'ig': 'igbo','is': 'icelandic','it': 'italian','iw': 'hebrew','ja': 'japanese','jw': 'javanese','ka': 'georgian','kk': 'kazakh',
             'km': 'khmer','kn': 'kannada','ko': 'korean','ku': 'kurdish (kurmanji)','ky': 'kyrgyz','la': 'latin','lb': 'luxembourgish',
             'lo': 'lao','lt': 'lithuanian','lv': 'latvian','mg': 'malagasy','mi': 'maori','mk': 'macedonian','ml': 'malayalam','mn': 'mongolian',
             'mr': 'marathi','ms': 'malay','mt': 'maltese','my': 'myanmar (burmese)','ne': 'nepali','nl': 'dutch','no': 'norwegian','ny': 'chichewa',
             'or': 'odia','pa': 'punjabi','pl': 'polish','ps': 'pashto','pt': 'portuguese','ro': 'romanian','ru': 'russian','sd': 'sindhi',
             'si': 'sinhala','sk': 'slovak','sl': 'slovenian','sm': 'samoan','sn': 'shona','so': 'somali','sq': 'albanian','sr': 'serbian',
             'st': 'sesotho','su': 'sundanese','sv': 'swedish','sw': 'swahili','ta': 'tamil','te': 'telugu','tg': 'tajik','th': 'thai','tw':'twi (ghana dialect)',
             'tl': 'filipino','tr': 'turkish','ug': 'uyghur','uk': 'ukrainian','ur': 'urdu','uz': 'uzbek','vi': 'vietnamese','xh': 'xhosa',
             'yi': 'yiddish','yo': 'yoruba','zh-cn': 'chinese (simplified)','zh-tw': 'chinese (traditional)','zu': 'zulu'}
             
#After some side preprocessing (removing inactive locals etc...), hardcode a list of Faker local codes. tw_GH stands for Twi, Ghanaian I figured...
FAKERLOCALS = {'ar_AA': 'Arabic (Egypt)','ar_PS': 'Arabic (Palestine)','ar_SA': 'Arabic (Saudi Arabia)','bg_BG': 'Bulgarian','cs_CZ': 'Czech','de_AT': 'German (Austria)',
               'de_CH': 'German (Switzerland)','de_DE': 'German','dk_DK': 'Danish','el_GR': 'Greek','en_AU': 'English (Australia)','en_CA': 'English (Canada)',
               'en_GB': 'English (Great Britain)','en_IN': 'English (India)','en_NZ': 'English (New Zealand)','en_US': 'English (United State)','es_ES': 'Spanish (Spain)',
               'es_MX': 'Spanish (Mexico)','et_EE': 'Estonian','fa_IR': 'Persian (Iran)','fi_FI': 'Finnish','fr_CH': 'French (Switzerland)','fr_FR': 'French',
               'fr_QC': 'French (Quebec)','he_IL': 'Hebrew (Israel)','hi_IN': 'Hindi','hr_HR': 'Croatian','hu_HU': 'Hungarian','hy_AM': 'Armenian',
               'id_ID': 'Indonesia','it_IT': 'Italian','ja_JP': 'Japanese','ka_GE': 'Georgian (Georgia)','ko_KR': 'Korean','lt_LT': 'Lithuanian','lv_LV': 'Latvian',
               'ne_NP': 'Nepali','nl_NL': 'Dutch (Netherlands)','no_NO': 'Norwegian','pl_PL': 'Polish','pt_BR': 'Portuguese (Brazil)','pt_PT': 'Portuguese (Portugal)',
               'ro_RO': 'Romanian','ru_RU': 'Russian','sl_SI': 'Slovene','sv_SE': 'Swedish','ta_IN': 'Tamil (India)','th_TH': 'Thai (Thailand)','tr_TR': 'Turkish',
               'tw_GH': 'Twi (Ghana)','uk_UA': 'Ukrainian','zh_CN': 'Chinese (China)','zh_TW': 'Chinese (Taiwan)'}

#Some characters we do not want names to includes, for example names can be composed with "/"
EXCLUDECHARS = ["(",")",".",":",";","&","/","0","8","["]

def produceNamesWithInfoDataset():
    """
    Use the package Faker to generate names accordind to gender, locality, and first vs. last.
    
    Arguments:
    None

    Returns: 
    namesWithInfo -- a dataframe containing a list of names and information on gender, locality and first vs. last n
    """
    #we use Faker to generate fake names, and stop when all possible names have been generated.
    #Unfortunately it was not possible to get a full list of names for every locals programatically.
    codesFakerLocal = list(FAKERLOCALS.keys())
    namesFakerLocal = list(FAKERLOCALS.values())
    Faker.seed(123)
    
    iterMax = 2000000
    patience = 4
    
    namesDict = dict()
    for code in codesFakerLocal:
        namesLocal = dict()
        fake = Faker(code)
        #different kind of name generators (first, last, male, female name)
        nameGenerators = dict(FNFemale = fake.first_name_female, FNMale = fake.first_name_male,LN = fake.last_name)
        for kind,generator in nameGenerators.items():
            #print(kind," ,", code) #debug
            namesLocalGenderFstvsLst = []
            nuniquePrev = 0
            countSame = 0
            iterNb = 0
            while countSame < patience and iterNb < iterMax:
                iterNb += 1
                #Faker will generate more names by making composite names (e.g. Smith-Doe).
                namesLocalGenderFstvsLst += generator().split("-")
                if iterNb % 5000 == 0: # check every 5000 iterations how much new names we are getting
                    nunique = len(set(namesLocalGenderFstvsLst))
                    if nuniquePrev == nunique : 
                        countSame += 1 
                    nuniquePrev = nunique
            namesLocal[kind] = list(set(namesLocalGenderFstvsLst))
        namesDict[code] = namesLocal
    
    #form a data frame 
    correspGender = dict(FNFemale = "Female", FNMale = "Male",LN = "Last")
    correspFirstVsLast = dict(FNFemale = "Firstname", FNMale = "Firstname",LN = "Lastname")

    names = []
    locality2 = []
    firstVsLast = []
    gender = []
    for locality,namesLocalDict in namesDict.items():
        for genderFstvsLst,namesLocal in namesLocalDict.items():
            #print(locality," ", genderFstvsLst," ",len(names)) #debug
            names += namesLocal
            locality2 += [locality]*len(namesLocal)
            gender += [correspGender[genderFstvsLst]]*len(namesLocal)
            firstVsLast += [correspFirstVsLast[genderFstvsLst]]*len(namesLocal)

    locality2Names = [FAKERLOCALS[key] for key in locality2]

    #a higher level for locality (more like language zone)
    locality1 = [local.split("_")[0] if local not in ["zh_TW","zh_CN"] else local for local in locality2]
    locality1Names = [LANGUAGES[key]  if key not in ["zh_TW","zh_CN"] else LANGUAGES[key.lower().replace("_","-")] for key in locality1]

    #we do not want to recognize names because they are capitalized
    names = [name.lower() for name in names]

    namesWithInfo = pd.DataFrame(dict(names=names,firstVsLast = firstVsLast,gender = gender,
                                      locality1=locality1,locality1Names=locality1Names,
                                      locality2=locality2,locality2Names=locality2Names))

    #removes names containing characters we do not want to include
    namesWithInfo = namesWithInfo.loc[[all(exChar not in l for exChar in EXCLUDECHARS) for l in namesWithInfo.names],:]
    #also remove empty names
    namesWithInfo = namesWithInfo.loc[[len(l) != 0 for l in namesWithInfo.names],:]
    #reset index because removing rows messed up the index
    namesWithInfo.reset_index(drop=True,inplace=True)

    return namesWithInfo

def predictionPipeline(name,classes,tokenizer,model,representation,inputLen):
  '''
  A function taking as input a name and building blocks of a prediction pipeline, 
  and returning as a dictionnary predicted features: locality, firstVsLastName, femaleVsMale,
  as well as an embedding vector.

  Parameters:
    name (str): a name to be processed.
    classes (list(str)): a list of the names of the classes predicted by the model.
    tokenizer (keras tokenizer): a tokenizer with a vocabulary, trained on the same train set as the model.
    model (keras model): a trained model for names features inference.
    representation (keras model): a model output embeddings for the name.
    inputLen (int): the model's inputs length.

  Returns:
    namesFeaturesDict (dict): a dictionnary with predicted features: locality, firstVsLastName, femaleVsMale
  '''
  name = name.lower()

  # tokenize names with the tokenizer
  sequence = tokenizer.texts_to_sequences([name])
  sequence = pad_sequences(sequences=sequence, maxlen=inputLen, padding='post')

  predictions = model.predict(sequence).ravel()

  nameFeaturesDict = dict()

  nameFeaturesDict["firstVsLastName"] = dict(firstName=predictions[1],lastName=predictions[3])
  nameFeaturesDict["femaleVsMale"] = dict(female=predictions[0],male=predictions[4])  
  nameFeaturesDict["locality"] = dict(zip(classes[5:],predictions[5:]))

  #representation
  repr = representation.predict(x=sequence)
  norm = np.linalg.norm(repr,axis=1).reshape(-1,1)
  repr = repr/norm

  nameFeaturesDict["representation"] = repr

  return nameFeaturesDict

#this method probably exists in nltk
def getCharsToIntMapping(names):
    """
    Output a mapping between characters of the vocabulary of the input list "names" and integers.
    The characters are sorted.
    
    Arguments:
    names -- a list of names from which we would like to extract a vocabulary.
    
    Returns: 
    charsToIntMapping -- a dictionnary containing a mapping between characters of the vocabulary of "names" and integers.
    """
    #the vocabulary
    chars = sorted(list(set([char for name in names for char in name])))
    charsNumb = len(chars)
    charsToIntMapping = dict(zip(chars,range(charsNumb)))
    return charsToIntMapping
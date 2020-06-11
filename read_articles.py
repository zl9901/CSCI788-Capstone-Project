


import json as js
import os
import xlrd

def read_articles():
    i = -1
    # input the number here
    ref = 28322
    """
    walk through all the files under specific directory
    """
    for dirname, _, filenames in os.walk('C:/PythonWorkspace/document_parses/pdf_json'):
        for filename in filenames:
            #print(os.path.join(dirname, filename))

            i += 1
            if i!=ref:
                continue


            """
            only load json files
            """
            if filename.split(".")[-1] == "json":

                f = open(os.path.join(dirname, filename))
                j = js.load(f)
                f.close()

                """
                some articles contain abstract while others not 
                """
                try:
                    abstract_text = ' '.join([x['text'] for x in j['abstract']])
                except:
                    abstract_text = ''


                cnt=0
                name = ''
                # d represents each individual dictionary
                # j['body_text'] is a python list which contains many dictionaries

                if cnt<1:
                    for dic in j['metadata']['authors']:
                        name += dic['first'] + ' ' + dic['last']
                        name += ', '
                    name = name[:-2]
                    cnt+=1



                """
                body text and abstract consist the whole body text
                """
                body_text = ' '.join(x['text'] for x in j['body_text'])
                body_text += ' ' + abstract_text

                # use i to print
                if i==ref:
                    print('authors are ' + name)
                    if abstract_text:
                        print(abstract_text)
                    else:
                        print(body_text)
                # use i to read specific article
                    exit()

    return

def extract_columns():

    index=[]
    loc = ('labels.xlsx')

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # return labels and number of the paper
    for i in range(1,sheet.nrows):
        index.append(int(sheet.cell_value(i, 3)))
    return index

# read_articles()
# read_articles method will exit the whole program in the middle process
print(len(set(extract_columns())))



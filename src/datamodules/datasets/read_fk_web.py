import urllib3
from bs4 import BeautifulSoup
import pdb
import yaml
import json 

org_url = 'https://www.forsakringskassan.se/'

result = []
qa_result = []
free_text_result = []

def clean(text, keep_line=False):

    if not keep_line:
        text = text.replace('\n','')
    text = text.replace('\xad', '_')
    text = text.replace('\xa0', ' ')
    text = text.replace(':', '')
    return text

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '\n\n', text)

def get_detail_page(content):

    # remove table 
    for table in content.select('table'):
        table.decompose()

    # remove video
    for video in content.select('div[class=fk-video]'):
        video.decompose()

    return clean(remove_html_tags(str(content)), keep_line=True)
        
def get_links_info(url,name='FK'):

    http_pool = urllib3.connection_from_url(url)
    r = http_pool.urlopen('GET', url)
    soup = BeautifulSoup(r.data.decode('utf-8'), 'html.parser')

    contents = soup.select('div[class*="startbox"]')

    if contents != None and len(contents) != 0:

        for content in contents:
        
            temp_dict = {}
            temp_cat = content.find('h2')
            temp_des = content.find('p')
            temp_link = temp_cat.find('a').get('href')
            temp_link = temp_link.replace('/wps/portal/', org_url)
            temp_dict['category'] = name + '-' + clean(temp_cat.text)
            temp_dict['content'] = clean(temp_des.text)
            temp_dict['link'] = temp_link
            result.append(temp_dict)

    else:

        sections = soup.find_all(['section'])

        for section in sections:
            temp_text = section.select('div[class="wpthemeControlBody"]')

            title_text = temp_text[0].find('p', {'class': 'ingress'})

            temp_result = dict()
            if title_text != None and len(title_text) != 0:
                temp_result['category'] = name
                temp_result['title'] = clean(title_text.text)
                text = get_detail_page(temp_text[0])
                temp_result['content'] = text
                free_text_result.append(temp_result)
    
        qa_contents = soup.select('div[class="fragor-svar"]')

        if qa_contents != None and len(qa_contents) > 0:
            qa_contents = qa_contents[0].find_all('div', {'class': ['fraga-text', 'content']})
            for index, content in enumerate(qa_contents):
                if (index % 2) == 0:
                    temp_dict = dict()
                    temp_dict['category'] = name
                    temp_dict['question'] = clean(content.text)
                else:
                    temp_dict['answer'] = clean(content.text)
                    qa_result.append(temp_dict)

get_links_info(org_url)

for i in result:
    if len(i['category'].split('-')) == 2:
        temp_url = i['link']
        temp_name = i['category']
        get_links_info(temp_url, temp_name)

for i in result:
    if len(i['category'].split('-')) == 3:
        temp_url = i['link']
        temp_name = i['category']
        get_links_info(temp_url, temp_name)

for i in result:
    if len(i['category'].split('-')) == 4:
        temp_url = i['link']
        temp_name = i['category']
        get_links_info(temp_url, temp_name)

with open('links.json', 'w', encoding='utf8') as file:
    json.dump(result, file, ensure_ascii=False)


with open('question_answer.json', 'w', encoding='utf8') as file:
    json.dump(qa_result, file, ensure_ascii=False)


with open('site_content.json', 'w', encoding='utf8') as file:
    json.dump(free_text_result, file, ensure_ascii=False)


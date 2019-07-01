"""
preprocess-twitter.py

python preprocess-twitter.py "Some random text with #hashtags, @mentions and http://t.co/kdjfkdjf (links). :)"

Script for preprocessing tweets by Romain Paulus
with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
"""

import json
import re
import emoji
from collections import OrderedDict
from nltk.tokenize import TweetTokenizer

# import spacy
# nlp = spacy.load('en_core_web_sm')
#
# def spacy_tokenizer(sentence):
#     return [tok.text for tok in nlp.tokenizer(sentence)]

tweet_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)


def sentence_split(sentence):
    return tweet_tokenizer.tokenize(sentence)


FLAGS = re.MULTILINE | re.DOTALL


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + hashtag_body.split(r"(?=[A-Z])"))
    return result


def allcaps(text):
    text = text.group()
    return text + " <allcaps>"  # add .lower() if want lowercase


def repeatchar(text):
    text = text.group()
    return text + " <repeat>"


def emojis(text):
    text = text.encode('utf-16', 'surrogatepass').decode('utf-16')
    for word in text:
        if word in emoji.UNICODE_EMOJI:
            emoji_desc = emoji.demojize(word)
            # plain_word = re.sub(r":\s?([\S]+)\s?:", r"<\1> ", emoji_desc)
            plain_word = re.sub(r"[^a-z]", r" ", emoji_desc)
            text = re.sub(word, plain_word, text)
    return text


def tokenize(text):
    sp_tokens = {'<url>', '<user>', '<smile>', '<lol>', '<sad>', '<neutral>', '<heart>', '<number>', '<repeat>',
                 '<elong>', '<hashtag>', '<allcaps>', '<person>', '<location>', '<amount>', '<time>', '<city>',
                 '<link>', '<score>', '<organization>', '<date>', '<data>', '<month>', '<department>', '<name>',
                 '<number>', '<day>', '<confirmation>', '<visa>', '<exam>', '<place>', '<park>', '<state>', '<bar>',
                 '<course>', '<event>', '<price>', '<year>', '<diagram>', '<action>', '<restaurant>', '<project>',
                 '<status>', '<measurement>', '<objects>', '<job>', '<profession>', '<program>', '<locastion>',
                 '<title>', '<center>', '<website>', '<street>', '<tax>', '<company>', '<country>', '<movie>',
                 '<coordinates>', '<wildlife>', '<museum>', '<group>', '<sight>', '<show>', '<avenue>', '<shopping>',
                 '<club>', '<casino>', '<subject>', '<test>', '<ranking>'}
    sp_t_map = {f'< {spt[1:-1]} >': spt for spt in sp_tokens}
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    text = text.lower()

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"/n", " ")
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"@\w+", "<user>")
    # text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    # text = re_sub(r"{}{}p+".format(eyes, nose), "<lol>")
    # text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sad>")
    # text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutral>")
    # text = re_sub(r"<3", "<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    # text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    # text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    # replace person and location
    text = text.replace("person_<number>", "<person>")
    text = text.replace("location_<number>", "<location>")
    text = text.replace("amount_<number>", "<amount>")
    text = text.replace("time_<number>", "<time>")
    text = text.replace("city_<number>", "<city>")
    text = text.replace("link_<number>", "<link>")
    text = text.replace("score_<number>", "<score>")
    text = text.replace("organization_<number>", "<organization>")
    text = text.replace("date_<number>", "<date>")
    text = text.replace("data_<number>", "<data>")
    text = text.replace("month_<number>", "<month>")
    text = text.replace("department_<number>", "<department>")
    text = text.replace("name_<number>", "<name>")
    text = text.replace("ame_<number>", "<name>")
    text = text.replace("number_<number>", "<number>")
    text = text.replace("days_<number>", "<day>")
    text = text.replace("day_<number>", "<day>")
    text = text.replace("confirmation_<number>", "<confirmation>")
    text = text.replace("visa_<number>", "<visa>")
    text = text.replace("exam_<number>", "<exam>")
    text = text.replace("place_<number>", "<place>")
    text = text.replace("park_<number>", "<park>")
    text = text.replace("state_<number>", "<state>")
    text = text.replace("bar_<number>", "<bar>")
    text = text.replace("course_<number>", "<course>")
    text = text.replace("event_<number>", "<event>")
    text = text.replace("price_<number>", "<price>")
    text = text.replace("year_<number>", "<year>")
    text = text.replace("diagram_<number>", "<diagram>")
    text = text.replace("action_<number>", "<action>")
    text = text.replace("restaurant_<number>", "<restaurant>")
    text = text.replace("project_<number>", "<project>")
    text = text.replace("status_<number>", "<status>")
    text = text.replace("measurement_<number>", "<measurement>")
    text = text.replace("objects_<number>", "<objects>")
    text = text.replace("job_<number>", "<job>")
    text = text.replace("profession_<number>", "<profession>")
    text = text.replace("program_<number>", "<program>")
    text = text.replace("locastion_<number>", "<locastion>")
    text = text.replace("title_<number>", "<title>")
    text = text.replace("center_<number>", "<center>")
    text = text.replace("website_<number>", "<website>")
    text = text.replace("street_<number>", "<street>")
    text = text.replace("tax_<number>", "<tax>")
    text = text.replace("country_<number>", "<country>")
    text = text.replace("company_<number>", "<company>")
    text = text.replace("movie_<number>", "<movie>")
    text = text.replace("coordinates_<number>", "<coordinates>")
    text = text.replace("wildlife_<number>", "<wildlife>")
    text = text.replace("museum_<number>", "<museum>")
    text = text.replace("group_<number>", "<group>")
    text = text.replace("sight_<number>", "<sight>")
    text = text.replace("show_<number>", "<show>")
    text = text.replace("avenue_<number>", "<avenue>")
    text = text.replace("club_<number>", "<club>")
    text = text.replace("shopping_<number>", "<shopping>")
    text = text.replace("casino_<number>", "<casino>")
    text = text.replace("subject_<number>", "<subject>")
    text = text.replace("test_<number>", "<test>")
    text = text.replace("ranking_<number>", "<ranking>")

    # find emoji
    text = emojis(text)

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    # text = re_sub(r"([A-Z]){2,}", allcaps)
    # text = re_sub(r"(\w)\1{2,}", repeatchar)
    # text = re_sub(r"(\w)\1{2,}(\S*)\b", r"\1\2 <repeat>")
    try:
        # remove unicode
        text = re.sub(r"\u0092|\x92", "'", text)
        text = text.encode("utf-8").decode("ascii", "ignore")
    except:
        pass
    # split with punctuations
    text = re_sub(r"([^A-Za-z0-9\_]+)", r" \1 ")
    text = sentence_split(text)
    # remove extra whitespace
    # text = " ".join(text.split())
    text = " ".join(text)
    for k, v in sp_t_map.items():
        text = text.replace(k, v)
    return text

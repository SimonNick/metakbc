#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests

root = "https://download.bio2rdf.org/files/"

def crawl(url: str):
    print(url)
    r = requests.get(url)
    d = r.json()
    for e in d:
        path = url.replace("https://download.bio2rdf.org/", "")
        
        name = e["name"]
        _type = e["type"]
        
        if _type == "directory":
            if not os.path.exists(path + "/" + name):
                os.makedirs(path + "/" + name)
            crawl(url + "/" + name)
        else:
            r2 = requests.get(url + "/" + name)
            with open(path + "/" + name, 'wb') as f:
                f.write(r2.content)
    return

crawl(root)

import httplib2
import json
import re

def get_dests():

    h = httplib2.Http('.cache')

    (res, page) = h.request('http://parkpgh.org', headers={'cache-control':'no-cache'})
    page        = page.decode('utf-8')
    expr1       = re.compile( 'park.lots = \[\{[^\;]+' )
    expr2       = re.compile( 'park.destinations = \[\{[^\;]+' )
    info1       = re.findall(expr1, page)
    info2       = re.findall(expr2, page)
    garas       = json.loads(info1[0][12:])
    dests       = json.loads(info2[0][20:])

    for k in range(len(garas)) :
        if garas[k]['status']['percent_available'] > 0 :
            ac_sp   = garas[k]['status']['actual_spaces']
            pe_av   = garas[k]['status']['percent_available'] / 100
            cap     = ac_sp / pe_av
        else :
            cap = 1
        
        garas[k]['cap']         = int(cap)
        garas[k]['cfcc']        = 1
        garas[k]['lanes']       = 0
        garas[k]['light']       = 0
        garas[k]['garage']      = 1
        garas[k]['destination'] = 0

        del garas[k]['status']


    for k in range(len(dests)) :
        dests[k]['cap']         = 0        
        dests[k]['cfcc']        = 1
        dests[k]['lanes']       = 0
        dests[k]['light']       = 0
        dests[k]['garage']      = 0
        dests[k]['destination'] = 1

    return garas, dests



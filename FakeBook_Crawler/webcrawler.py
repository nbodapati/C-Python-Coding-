import sys 
import socket 
import re
import ssl

uname=sys.argv[1]
passwd=sys.argv[2]


request = b"""\
GET /accounts/login/ HTTP/1.1\r\n"""

headers=b"""\
Host:elsrv2.cs.umass.edu\r\n\r\n
"""

payload = (request+headers)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("elsrv2.cs.umass.edu", 80))
s.sendall(payload)

results=""
while True:
    new = s.recv(4096)
    n=str(new)
    results=results+n
    if not new:
      s.close()
      break

results=re.findall(r'Set-Cookie:(.*?);',results)
csrftoken_=None
sessionid=None

for i in range(len(results)):
    r=results[i]
    hn,hv=r.split("=")      
    if(hn.strip()=="csrftoken"):
       csrftoken_=csrfmiddlewaretoken_=hv
    elif(hn.strip()=="sessionid"):
       sessionid_=hv


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("elsrv2.cs.umass.edu", 80))

body="""\
username={username_}&password={password_}&csrfmiddlewaretoken={token}"""

body_bytes=body.format(
token=csrftoken_,
username_=uname,
password_=passwd).encode()

payload = '''POST /accounts/login/?next=/fakebook/ HTTP/1.1
Content-Type: application/x-www-form-urlencoded
Content-Length: %d
Cookie: csrftoken=%s; sessionid=%s
Host: elsrv2.cs.umass.edu
Connection:keep-alive

username=%s&password=%s&csrfmiddlewaretoken=%s
''' % (len(body_bytes),csrftoken_, sessionid_,uname,passwd,csrftoken_)


s.sendall(payload.encode())

results=""
while True:
    new = s.recv(4096)
    n=str(new)#.replace("\r\n","")
    results=results+n
    if not new:
      s.close()
      break

#print(results)

location=re.findall(r'.*Location: http://elsrv2.cs.umass.edu(.*?)\r\n',results)
location=location[0].strip()

sessionid=re.findall(r'.*sessionid=(.*?);',results)[0]
#print("sessionid= ",sessionid)

parsed_list=[]
to_parse=[location]

def load_hyperlinks(location):
    global sessionid,csrftoken_,parsed_list
    move_on=False
    results=""
    while (move_on!=True):
          request = """\
          GET %s  HTTP/1.1\r\n"""%location

          headers="""Cookie:csrftoken=%s;sessionid=%s\r\n"""%(csrftoken_,sessionid)
          headers=headers+"Host:elsrv2.cs.umass.edu\r\n\r\n"
          payload = (request+headers)
          #print("Payload: ",payload)
          s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          s.connect(("elsrv2.cs.umass.edu", 80))
          s.sendall(payload.encode())
          new = s.recv(4096)
          n=str(new)
          #print("received code: ",n[9:12])
          if(n[9:12]=="200" or  n[9:12]=="404" or n[9:12]=="403"):
             move_on=True
             continue
          elif(n[9:12]=="500"):
             move_on=False
          elif( n[9:12]=="301"):
              while True:
                  new = s.recv(4096)
                  n=str(new)#.replace("\r\n","")
                  results=results+n
                  if not new:
                     s.close()
                     break

                  location=re.findall(r'.*Location: http://elsrv2.cs.umass.edu(.*?)\r\n',results)
                  location=location[0].strip()
                  if(location not in parsed_list):
                     parsed_list.append(location)
                     #print("location 301 error: ",location)
                     continue
                  else:
                     #print("Already parsed!")
                     return [] 
                                   
    while True:
          new = s.recv(4096)
          n=str(new)
          results=results+n
          if not new:
             s.close()
             break

    #print(results)
    p = results.find('\r\n\r\n')
    if p >= 0:
       parsed_html=(results[p+4:])
       #print("parsed_html: ",parsed_html)
       find_next_links=re.findall(r'<a href\="(.*?)">',parsed_html)
       #print("NExt links: ",find_next_links)
      
       secret_flag=re.match(r'<h2 class=\'secret_flag\' style=\"color:red\">(.*?)</h2>',parsed_html)  
       secret_flag=parsed_html.find("secret_flag")#re.match(r'<.*secret_flag.*>',parsed_html)
       #print("secret_flag: ",secret_flag)
                
       if(secret_flag!=-1):
         #print("**********************secret flag:",secret_flag)
         #print("*************parsed_html:",parsed_html) 
         flag=parsed_html.find("FLAG: ")
         #print(flag)
         print(parsed_html[flag+6:flag+70])
       return find_next_links
    else:
       #print("No hyperlinks!") 
       return []  

def web_crawl():
    global parsed_list,to_parse
    #print("Location to crawl to: ",location)
    to_parse2=[]
    for next_loc in to_parse:
          #print("next_loc: ",next_loc,next_loc[:10])
           page_hyperlinks=load_hyperlinks(next_loc)             
           parsed_list.append(next_loc)
           for ph in page_hyperlinks:
               if(ph not in parsed_list and  ph not in to_parse2 and ph not in to_parse and ph[:10]=='/fakebook/'):
                  to_parse2.append(ph) 
    to_parse=to_parse2
    #print("PArsed list:: ",parsed_list)
    #print("Ti parse list:: ",to_parse)        

while(to_parse!=[]):
     #print("Calling webcrawl...",len(parsed_list),len(to_parse))
     web_crawl()   


    




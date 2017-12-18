import sys
import socket

def create_client(host="10.0.2.15",port=53):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')
        try:
           s.connect((host, port))
        except socket.error as msg:
           print 'Connect failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
           sys.exit()
        data="falun.com"
        s.send(data)
        print("sent data",data)
        count=0
        while(1):
            resp=s.recv(1024) #blocking call
            print(resp)
            count+=1
            if(count==2):
                break
        s.close()

if __name__=='__main__':
    create_client()

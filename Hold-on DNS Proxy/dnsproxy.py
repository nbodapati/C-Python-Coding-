from scapy.all import *
import time
import socket
import sys

try:
    import threading as _threading
except ImportError:
    import dummy_threading as _threading

#Poll the server for nin-sensitive names to get an idea over the network
#architecture.
eslrv2="128.119.243.134"
google="8.8.8.8" #public,free primary dns server/
uncensored="91.239.100.100"
dns_watch="84.200.69.80"

is_packet_validated=False
is_dnssec_enabled=False
destination=None
timer_obj=None

ttl_threshold=2
rtt_threshold=0.8
count_pkts=0
expected_rtt=None
expected_ttl=None
start=None
end=None
hperiod=None
st=None
en=None

def expected_values():
    global expected_rtt,expected_ttl
    ttls=[]
    rtts=[]
    for i in range(100):
        start_=time.time()
        a,u=sr(IP(dst=eslrv2)/UDP(dport=5300)/DNS(rd=1,qd=DNSQR(qname="www.google.com")))
        end_=time.time()
        rtts.append(a[0][1].time -start_)
        ttls.append(a[0][1].ttl)
        print("Iteration: ",i,"RTT: ",(end_-start_),"TTL: ",a[0][1].ttl)

    expected_ttl=min(ttls)#sum(ttls)/len(ttls) #min(ttls)
    expected_rtt=sum(rtts)/len(rtts) #min(rtts)
    print("Expeted TTL: ",expected_ttl)
    print("Expected RTT: ",expected_rtt)

def is_ttl_valid(ttl):
    global expected_rtt,expected_ttl,ttl_threshold,rtt_threshold
    if(ttl<=(expected_ttl+ttl_threshold) and ttl>=(expected_ttl-ttl_threshold)):
        return True
    else:
        return False

def is_rtt_valid(rtt):
    global expected_rtt,expected_ttl,ttl_threshold,rtt_threshold
    if(rtt>=(0.5*expected_rtt)):
        return True
    else:
        return False

def is_packet_valid(ttl,rtt):
    if(is_ttl_valid(ttl) and is_rtt_valid(rtt)):
        return True
    else:
        return False

#To create a timer for hold-on period in a new thread.
#The thread exits when the timer is off and checks if a valid packet is received
#within the time period.
#Timer is stopped or thread is killed(?) when a valid packet is found.
class Timer(object):
    def __init__(self,hold_on):
        self.hold_on_period=hold_on
        self.timer=self.create_timer()

    def create_timer(self):
        timer=_threading.Timer(self.hold_on_period,self.callback)
        print("Created Timer...")
        return timer

    def start_timer(self):
        print("Starting timer...",self.timer)
        self.timer.start()

    def stop_timer(self):
        print("Stopping the timer..")
        if(self.timer.is_alive()):
            self.timer.cancel()

    def is_alive(self):
        return self.timer.is_alive()

    def callback(self):
        #call this when the timer ends.
        print("Callback funcition after timer ends..")
        global is_packet_validated
        if(is_packet_validated==False):
            print("Have not received a response from the dns derver yet.")
        self.stop_timer()

def start_timer(hold_on_period):
    timer_obj=Timer(hold_on_period)
    timer_obj.start_timer()
    return timer_obj

#Validate each packet sniffed.
def validate_packet(packet):
    global start,end,destination,is_packet_validated,st,en,count_pkts
    #print(packet[0][0],packet[0][1])
    print(packet[IP].dst,packet[IP].src,packet.time-start)

    rtt=packet.time-start
    ttl=packet[0][1].ttl
    print("TTL: ",ttl,"RTT: ",rtt)
    count_pkts+=1

    if(count_pkts==2 and is_packet_valid(ttl,rtt)):
    #if(is_packet_valid(ttl,rtt)):
        is_packet_validated=True
        count_pkts=0
        print("valid packet rx..!!!")
    else:
        print("Invalid/Censored packet..!! or false positive(?)")

def stop_sniffer(packet):
    global timer_obj,is_packet_validated

    if(timer_obj.is_alive()==False):
        return True
    if(is_packet_validated==True):
        timer_obj.stop_timer()
        print("Stopping the timer..")
        return True
    else:
        return False

def lfilter(pack):
    global end
    end=time.time()
    dst="128.119.243.134";
    src="10.0.2.15"
    if(UDP in pack and IP in pack and pack[IP].dst == src and pack[IP].src==dst ):
        return True

def send_request(dst,qname,port=53):
    global start,destination,hperiod,st,en
    import subprocess
    destination=dst

    st=time.time()
    #dumpcap_args=["dumpcap",'-w','-','-P']
    #dumpcap=subprocess.Popen(args=dumpcap_args,stdout=subprocess.PIPE)
    en=time.time()

    start=time.time()
    send(IP(dst=dst)/UDP(dport=port)/DNS(rd=1,qd=DNSQR(qname=qname)))
    sniff(lfilter=lfilter, prn=validate_packet,stop_filter=stop_sniffer,timeout=hperiod)#,offline=dumpcap.stdout)


#This is on receiving a request.
def hold_on_proxy():
    global start,timer_obj,is_packet_validated,hperiod,count_pkts
    hold_on_periods=[15,20,25]
    for attempt in range(3):
        count_pkts=0
        print("Hold-on period..",hold_on_periods[attempt])
        hperiod=hold_on_periods[attempt]
        timer_obj=start_timer(hold_on_periods[attempt])
        send_request(eslrv2,qname="falun.com",port=5300)

        if(is_packet_validated==True):
            break


if __name__ =='__main__':
    expected_values()
    hold_on_proxy()
    #interact(mydict=globals(), mybanner="Scapy DNS Tester")

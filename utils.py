from email.mime.multipart import MIMEMultipart
from email.utils import COMMASPACE, formatdate
from email.mime.text import MIMEText
import smtplib
from datetime import datetime
import os

def email(text):
  server="127.0.0.1"
  msg = MIMEMultipart()
  send_from='refitt-planner@purdue.edu'
  send_to=['niharika.sravan@gmail.com']#,'bsubraya@purdue.edu']
  msg['From'] = send_from
  msg['To'] = COMMASPACE.join(send_to)
  msg['Date'] = formatdate(localtime=True)
  msg['Subject'] = 'Daily run exception'
  msg.attach(MIMEText(text))

  smtp = smtplib.SMTP(server)
  smtp.sendmail(send_from, send_to, msg.as_string())
  smtp.close()

def log(log):
  dir_path = os.path.dirname(os.path.realpath(__file__)) #FIXME: hardcoded
  log_str=datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' '+log+'\n'
  with open(dir_path+'/ZTF/daily_log','a') as f:
    f.write(log_str)


from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
from lxml import etree as E
import re, json

BASE = Path(r'C:\UNI\BA Final Project')
WORK = BASE / 'tmp/chapter10'
SRC = BASE / 'report/Project_Report_new.docx'
OUT = BASE / 'report/Project_Report_with_Chapter10.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
q = lambda name: '{'+W+'}'+name

def el(tag, **attrs):
    e=E.Element(q(tag))
    for k,v in attrs.items():e.set(q(k),str(v))
    return e

def setting(parent, tag, **attrs):
    e=parent.find(q(tag))
    if e is None:e=el(tag);parent.append(e)
    for k,v in attrs.items():e.set(q(k),str(v))
    return e

with ZipFile(SRC) as z:
    entries={n:z.read(n) for n in z.namelist()}
root=E.fromstring(entries['word/document.xml'])
body=root.find('w:body',NS)
styles=E.fromstring(entries['word/styles.xml'])
bookmark_id=max(int(x) for x in root.xpath('//w:bookmarkStart/@w:id',namespaces=NS))+1

def bookmark(p,name):
    global bookmark_id
    a=el('bookmarkStart',id=bookmark_id,name=name)
    b=el('bookmarkEnd',id=bookmark_id)
    p.insert(1,a);p.append(b);bookmark_id+=1

def add_runs(p,text,size=28,bold=False,heading=False):
    # Keep Latin names in explicit left-to-right runs, as used in the report.
    parts=[text] if heading else re.split(r'([A-Za-z][A-Za-z0-9@.-]*)',text)
    for part in parts:
        if not part:continue
        latin=bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9@.-]*',part))
        r=el('r');rp=el('rPr')
        rp.append(el('rFonts',ascii='Times New Roman' if latin else 'B Zar',hAnsi='Times New Roman' if latin else 'B Zar',cs='B Zar'))
        if bold:rp.extend([el('b'),el('bCs')])
        rp.append(el('color',val='000000'))
        rp.append(el('sz',val=(24 if size==28 and latin else size)))
        rp.append(el('szCs',val=size))
        rp.append(el('rtl',val='0' if latin else '1'))
        rp.append(el('lang',val='en-US' if latin else 'fa-IR',bidi='fa-IR'))
        r.append(rp);t=el('t');t.set('{http://www.w3.org/XML/1998/namespace}space','preserve');t.text=part;r.append(t);p.append(r)

def para(text,level=0):
    p=el('p');pp=el('pPr')
    if level:
        pp.append(el('pStyle',val=f'Heading{level}'))
        pp.append(el('keepNext'));pp.append(el('keepLines'))
        if level==1:pp.append(el('pageBreakBefore'))
    pp.append(el('widowControl'))
    pp.append(el('bidi'))
    pp.append(el('spacing',line=360,lineRule='auto',after=(80 if level!=3 else 60)))
    pp.append(el('jc',val='right' if level else 'both'))
    p.append(pp)
    add_runs(p,text,size=28 if level<=1 else (24 if level==2 else 20),bold=bool(level),heading=bool(level))
    return p

table_rows=[
    ['بخش و روش','شاخص‌های اصلی','برداشت از نتیجه'],
    ['خرید دوباره\nتقویت گرادیانی','ROC-AUC برابر ۰/۸۷۳\nF1 برابر ۰/۸۱۲\nبریر برابر ۰/۱۴۷','نتیجه مناسب‌تر در پیکربندی بررسی‌شده؛ وابسته به افق و نمونه آموزش'],
    ['درآمد محصول\nمدل مستقیم و دو مبنا','خطای نسبی وزنی\nمستقیم ۴۶ درصد\nبازگشتی ۴۶/۷ درصد\nمیانگین اخیر ۴۸/۷ درصد','کاهش خطای نسبی مدل مستقیم؛ برتری مبنای ساده در برخی معیارهای دیگر'],
    ['رتبه‌بندی درآمد\nهر سه روش','اشتراک بیست محصول برتر\n۱۶ مورد از ۲۰ مورد','تفاوت محدود روش‌ها در این معیار رتبه‌بندی'],
    ['ریزش مشتری\nمدل فعال','ROC-AUC برابر ۰/۸۸۲\nPR-AUC برابر ۰/۶۰۶\nبازخوانی ۰/۶۷۳','کاربرد برای اولویت‌بندی؛ همراه با ریزش‌های ازدست‌رفته و هشدار نادرست'],
    ['پیشنهاد محصول\nمدل ترکیبی','ROC-AUC برابر ۰/۶۲۶۸\nPR-AUC برابر ۰/۰۳۱۹\nنرخ مثبت آزمون ۱/۸۴ درصد','توان تفکیک محدود؛ کیفیت بیست پیشنهاد نهایی هنوز اندازه‌گیری نشده است'],
]

def results_table():
    cap=para('جدول ۱۰-۱ - خلاصه نتایج کمی ارزیابی مدل‌ها')
    pp=cap.find(q('pPr'));pp.insert(0,el('pStyle',val='DynamicTableCaption'))
    setting(pp,'keepNext');setting(pp,'keepLines');setting(pp,'spacing',before=100,after=80,line=240,lineRule='auto');setting(pp,'jc',val='center')
    for rp in cap.findall('w:r/w:rPr',NS):setting(rp,'sz',val=24);setting(rp,'szCs',val=24)
    bookmark(cap,'tbl_chapter10_1')
    tbl=el('tbl');pr=deepcopy(body[1672].find(q('tblPr')))
    setting(pr,'tblW',w=8778,type='dxa');setting(pr,'jc',val='center');setting(pr,'tblLayout',type='fixed')
    tbl.append(pr);grid=el('tblGrid');widths=[2100,3178,3500]
    for width in widths:grid.append(el('gridCol',w=width))
    tbl.append(grid)
    for ri,row in enumerate(table_rows):
        tr=el('tr');trp=el('trPr');trp.append(el('cantSplit'))
        if ri==0:trp.append(el('tblHeader'))
        tr.append(trp)
        for width,text in zip(widths,row):
            tc=el('tc');tcp=el('tcPr');tcp.append(el('tcW',w=width,type='dxa'))
            tcp.append(el('shd',val='clear',color='auto',fill='D9D9D9' if ri==0 else 'FFFFFF'))
            mar=el('tcMar')
            for side,value in [('top',100),('left',120),('bottom',100),('right',120)]:mar.append(el(side,w=value,type='dxa'))
            tcp.append(mar);tcp.append(el('vAlign',val='center'));tc.append(tcp)
            p=el('p');pp=el('pPr');pp.append(el('bidi'));pp.append(el('spacing',after=0,line=276,lineRule='auto'));pp.append(el('jc',val='center'))
            if ri==0:pp.append(el('keepNext'))
            p.append(pp)
            for j,line in enumerate(text.split('\n')):
                if j:r=el('r');r.append(el('br'));p.append(r)
                add_runs(p,line,size=24,bold=ri==0)
            tc.append(p);tr.append(tc)
        tbl.append(tr)
    return [cap,tbl]

nodes=[]
for block in (WORK/'chapter10.md').read_text(encoding='utf-8').strip().split('\n\n'):
    if block=='{{RESULTS_TABLE}}':nodes.extend(results_table());continue
    m=re.match(r'^(#{1,3}) (.+)$',block,re.S)
    if m:
        level=len(m[1]);p=para(m[2],level)
        bookmark(p,'chapter10_start' if level==1 else f'chapter10_heading_{len(nodes)}')
    else:p=para(block)
    nodes.append(p)

reference=body[1688]
assert ''.join(reference.xpath('.//w:t/text()',namespaces=NS)).strip('\u200f ')=='منابع'
for p in nodes:reference.addprevious(p)
entries['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(OUT,'w',compression=ZIP_DEFLATED) as z:
    for n,data in entries.items():z.writestr(n,data)
with ZipFile(SRC) as a, ZipFile(OUT) as b:
    changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
assert changed==['word/document.xml'],changed
(WORK/'build_info.json').write_text(json.dumps({'output':str(OUT),'inserted_nodes':len(nodes),'words':len((WORK/'chapter10.md').read_text(encoding='utf-8').split()),'changed_parts':changed},indent=2),encoding='utf-8')
print(str(OUT));print('Inserted nodes:',len(nodes));print('Source parts preserved except inserted document content')

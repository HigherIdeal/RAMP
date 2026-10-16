package com.higherideal.pdfcleaner;

import com.tom_roush.pdfbox.contentstream.operator.Operator;
import com.tom_roush.pdfbox.cos.COSArray;
import com.tom_roush.pdfbox.cos.COSBase;
import com.tom_roush.pdfbox.cos.COSDictionary;
import com.tom_roush.pdfbox.cos.COSName;
import com.tom_roush.pdfbox.cos.COSNumber;
import com.tom_roush.pdfbox.cos.COSString;
import com.tom_roush.pdfbox.pdfparser.PDFStreamParser;
import com.tom_roush.pdfbox.pdfwriter.ContentStreamWriter;
import com.tom_roush.pdfbox.pdmodel.PDDocument;
import com.tom_roush.pdfbox.pdmodel.PDPage;
import com.tom_roush.pdfbox.pdmodel.PDResources;
import com.tom_roush.pdfbox.pdmodel.common.PDStream;
import com.tom_roush.pdfbox.pdmodel.documentinterchange.markedcontent.PDPropertyList;
import com.tom_roush.pdfbox.pdmodel.graphics.PDXObject;
import com.tom_roush.pdfbox.pdmodel.graphics.form.PDFormXObject;
import com.tom_roush.pdfbox.pdmodel.graphics.optionalcontent.PDOptionalContentGroup;
import com.tom_roush.pdfbox.pdmodel.graphics.state.PDExtendedGraphicsState;
import com.tom_roush.pdfbox.pdmodel.interactive.annotation.PDAnnotation;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.security.MessageDigest;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** Object-level PDF watermark analyzer/remover. Pages are never rasterized. */
public final class PdfObjectCleaner {
    public enum Kind { ANNOTATION_WATERMARK, TAGGED_WATERMARK, OPTIONAL_CONTENT, TEXT, FORM_XOBJECT, LOW_ALPHA_BLOCK }

    public static final class Candidate {
        public final String key;
        public final Kind kind;
        public String label;
        public String detail;
        public int occurrences;
        public int pageCount;
        public int totalPages;
        public float minAlpha = 1f;
        public float maxAbsAngle = 0f;
        public float maxFontSize = 0f;
        public boolean defaultSelected;
        private final Set<Integer> pages = new HashSet<>();

        Candidate(String key, Kind kind, String label, String detail, int totalPages) {
            this.key = key; this.kind = kind; this.label = label; this.detail = detail; this.totalPages = totalPages;
        }
        void add(int page, float alpha, float angle, float fontSize) {
            occurrences++; pages.add(page); pageCount = pages.size();
            minAlpha = Math.min(minAlpha, alpha); maxAbsAngle = Math.max(maxAbsAngle, Math.abs(angle)); maxFontSize = Math.max(maxFontSize, fontSize);
        }
        public float coverage() { return totalPages <= 0 ? 0f : (float) pageCount / totalPages; }
        int score() {
            int s;
            switch (kind) {
                case ANNOTATION_WATERMARK:
                case TAGGED_WATERMARK: s = 100; break;
                case OPTIONAL_CONTENT: s = 75; break;
                case LOW_ALPHA_BLOCK: s = 65; break;
                case TEXT: s = 50; break;
                default: s = 40;
            }
            s += Math.round(coverage() * 20f);
            if (minAlpha <= 0.5f) s += 12; else if (minAlpha < 0.8f) s += 6;
            if (maxAbsAngle >= 10f) s += 8;
            if (maxFontSize >= 24f) s += 4;
            return Math.min(100, s);
        }
        public String displayText() {
            int s = score();
            String confidence = s >= 85 ? "높음" : (s >= 65 ? "중간" : "낮음");
            return label + "\n" + detail + " · " + pageCount + "/" + totalPages + "페이지 · 신뢰도 " + confidence;
        }
    }

    private static final class Occurrence {
        String key; Kind kind; int page; int start; int endExclusive; float alpha; float angle; float fontSize; String label; String detail; boolean annotation;
        Occurrence(String key, Kind kind, int page, int start, int endExclusive, float alpha, float angle, float fontSize, String label, String detail) {
            this.key = key; this.kind = kind; this.page = page; this.start = start; this.endExclusive = endExclusive; this.alpha = alpha; this.angle = angle; this.fontSize = fontSize; this.label = label; this.detail = detail;
        }
    }
    private static final class TextBlock { int start; float alpha = 1f; float fontSize; float angle; final StringBuilder text = new StringBuilder(); TextBlock(int start, float alpha){this.start=start;this.alpha=alpha;} }
    private static final class QBlock { int start; float minAlpha = 1f; boolean visible; QBlock(int start,float alpha){this.start=start;this.minAlpha=alpha;} }
    private static final class MarkBlock { int start; boolean watermark; String name; MarkBlock(int start,boolean watermark,String name){this.start=start;this.watermark=watermark;this.name=name;} }
    private static final class MarkInfo { boolean isWatermark; String name; }
    private PdfObjectCleaner() {}

    public static List<Candidate> analyze(File input) throws IOException {
        PDDocument doc = PDDocument.load(input);
        try {
            int total = doc.getNumberOfPages();
            LinkedHashMap<String,Candidate> map = new LinkedHashMap<>();
            for (int p=0;p<total;p++) {
                for (Occurrence o: scanPage(doc.getPage(p),p,total)) {
                    Candidate c = map.get(o.key);
                    if (c==null) { c = new Candidate(o.key,o.kind,o.label,o.detail,total); map.put(o.key,c); }
                    c.add(p,o.alpha,o.angle,o.fontSize);
                }
            }
            List<Candidate> out = new ArrayList<>();
            for (Candidate c: map.values()) {
                boolean keep;
                if (c.kind==Kind.ANNOTATION_WATERMARK || c.kind==Kind.TAGGED_WATERMARK) keep=true;
                else if (total==1) keep = c.kind==Kind.LOW_ALPHA_BLOCK || c.kind==Kind.OPTIONAL_CONTENT || (c.kind==Kind.TEXT && (c.minAlpha<0.85f || c.maxAbsAngle>=8f || c.maxFontSize>=20f));
                else {
                    float cov=c.coverage();
                    if (c.kind==Kind.FORM_XOBJECT) keep=cov>=0.50f;
                    else if (c.kind==Kind.TEXT) keep=cov>=0.40f && (c.minAlpha<0.90f || c.maxAbsAngle>=8f || c.maxFontSize>=20f);
                    else if (c.kind==Kind.LOW_ALPHA_BLOCK) keep=cov>=0.35f && c.minAlpha<0.80f;
                    else keep=cov>=0.35f;
                }
                if (keep) {
                    c.defaultSelected = (c.kind==Kind.ANNOTATION_WATERMARK || c.kind==Kind.TAGGED_WATERMARK) ||
                            (c.kind==Kind.LOW_ALPHA_BLOCK && c.coverage()>=0.70f && c.minAlpha<=0.50f) ||
                            (c.kind==Kind.TEXT && c.coverage()>=0.70f && (c.minAlpha<=0.60f || c.maxAbsAngle>=15f));
                    out.add(c);
                }
            }
            Collections.sort(out,new Comparator<Candidate>(){ @Override public int compare(Candidate a,Candidate b){ int s=Integer.compare(b.score(),a.score()); return s!=0?s:a.label.compareToIgnoreCase(b.label); }});
            return out;
        } finally { doc.close(); }
    }

    public static int remove(File input, File output, Set<String> selectedKeys) throws IOException {
        PDDocument doc=PDDocument.load(input); int removed=0;
        try {
            int total=doc.getNumberOfPages();
            for (int p=0;p<total;p++) {
                PDPage page=doc.getPage(p);
                List<Occurrence> occ=scanPage(page,p,total);
                boolean removeAnn=false;
                for (Occurrence o:occ) if (o.annotation && selectedKeys.contains(o.key)) {removeAnn=true;break;}
                if (removeAnn) {
                    List<PDAnnotation> anns=page.getAnnotations();
                    for (int i=anns.size()-1;i>=0;i--) {
                        COSName subtype=anns.get(i).getCOSObject().getCOSName(COSName.SUBTYPE);
                        if (subtype!=null && "Watermark".equalsIgnoreCase(subtype.getName())) {anns.remove(i);removed++;}
                    }
                }
                PDFStreamParser parser=new PDFStreamParser(page); parser.parse(); List<Object> tokens=parser.getTokens(); boolean[] drop=new boolean[tokens.size()]; boolean any=false;
                for (Occurrence o:occ) {
                    if (o.annotation || !selectedKeys.contains(o.key)) continue;
                    int s=Math.max(0,Math.min(tokens.size(),o.start)); int e=Math.max(s,Math.min(tokens.size(),o.endExclusive));
                    for (int i=s;i<e;i++) drop[i]=true;
                    any=true; removed++;
                }
                if (any) {
                    List<Object> kept=new ArrayList<>(tokens.size()); for (int i=0;i<tokens.size();i++) if(!drop[i]) kept.add(tokens.get(i));
                    PDStream newContent=new PDStream(doc); OutputStream os=newContent.createOutputStream(COSName.FLATE_DECODE);
                    try { new ContentStreamWriter(os).writeTokens(kept); } finally { os.close(); }
                    page.setContents(newContent);
                }
            }
            doc.save(output); return removed;
        } finally { doc.close(); }
    }

    private static List<Occurrence> scanPage(PDPage page,int pageIndex,int totalPages) throws IOException {
        List<Occurrence> out=new ArrayList<>(); PDResources resources=page.getResources(); if(resources==null) resources=new PDResources();
        for(PDAnnotation ann:page.getAnnotations()) {
            COSName subtype=ann.getCOSObject().getCOSName(COSName.SUBTYPE);
            if(subtype!=null && "Watermark".equalsIgnoreCase(subtype.getName())) { Occurrence o=new Occurrence("ANNOTATION:WATERMARK",Kind.ANNOTATION_WATERMARK,pageIndex,-1,-1,1f,0f,0f,"확실한 Watermark annotation","/Subtype /Watermark"); o.annotation=true; out.add(o); }
        }
        PDFStreamParser parser=new PDFStreamParser(page); parser.parse(); List<Object> tokens=parser.getTokens(); int operandStart=0; float alpha=1f;
        Deque<Float> alphaStack=new ArrayDeque<>(); Deque<QBlock> qBlocks=new ArrayDeque<>(); Deque<MarkBlock> markBlocks=new ArrayDeque<>(); TextBlock textBlock=null;
        for(int i=0;i<tokens.size();i++) {
            Object token=tokens.get(i); if(!(token instanceof Operator)) continue;
            String op=((Operator)token).getName(); List<Object> args=tokens.subList(operandStart,i);
            if("q".equals(op)) { alphaStack.push(alpha); qBlocks.push(new QBlock(operandStart,alpha)); }
            else if("Q".equals(op)) {
                if(!qBlocks.isEmpty()) { QBlock b=qBlocks.pop(); if(b.visible && b.minAlpha<0.80f){ String key="ALPHA:"+hashTokens(tokens,b.start,i+1); out.add(new Occurrence(key,Kind.LOW_ALPHA_BLOCK,pageIndex,b.start,i+1,b.minAlpha,0f,0f,"저투명도 그래픽 블록",String.format(Locale.US,"opacity α≈%.2f",b.minAlpha))); } }
                alpha=alphaStack.isEmpty()?1f:alphaStack.pop();
            }
            else if("gs".equals(op)) {
                COSName gsName=firstName(args); if(gsName!=null) try { PDExtendedGraphicsState gs=resources.getExtGState(gsName); if(gs!=null){ Float ca=gs.getNonStrokingAlphaConstant(); Float CA=gs.getStrokingAlphaConstant(); float next=1f; if(ca!=null)next=Math.min(next,ca); if(CA!=null)next=Math.min(next,CA); alpha=next; for(QBlock b:qBlocks)b.minAlpha=Math.min(b.minAlpha,alpha); if(textBlock!=null)textBlock.alpha=Math.min(textBlock.alpha,alpha); } } catch(Exception ignored){}
            }
            else if("BT".equals(op)) textBlock=new TextBlock(operandStart,alpha);
            else if("Tf".equals(op) && textBlock!=null) { if(args.size()>=2 && args.get(args.size()-1) instanceof COSNumber) textBlock.fontSize=((COSNumber)args.get(args.size()-1)).floatValue(); }
            else if("Tm".equals(op) && textBlock!=null) { if(args.size()>=6 && args.get(0) instanceof COSNumber && args.get(1) instanceof COSNumber){ float a=((COSNumber)args.get(0)).floatValue(); float b=((COSNumber)args.get(1)).floatValue(); textBlock.angle=(float)Math.toDegrees(Math.atan2(b,a)); } }
            else if(isTextShow(op) && textBlock!=null) { appendText(args,textBlock.text); textBlock.alpha=Math.min(textBlock.alpha,alpha); markVisible(qBlocks); }
            else if("ET".equals(op) && textBlock!=null) { String raw=normalizeText(textBlock.text.toString()); if(!raw.isEmpty()){ String shown=shorten(raw,44); String key="TEXT:"+raw; String detail=String.format(Locale.US,"텍스트 '%s' · α≈%.2f · %.0f° · %.0fpt",shown,textBlock.alpha,textBlock.angle,textBlock.fontSize); out.add(new Occurrence(key,Kind.TEXT,pageIndex,textBlock.start,i+1,textBlock.alpha,textBlock.angle,textBlock.fontSize,"반복 텍스트 후보: "+shown,detail)); } textBlock=null; }
            else if("Do".equals(op)) { markVisible(qBlocks); COSName name=firstName(args); if(name!=null) try { PDXObject xo=resources.getXObject(name); if(xo instanceof PDFormXObject){ PDFormXObject form=(PDFormXObject)xo; String hash=sha256(readAll(form.getContents())); String formText=extractRawText(form); String shown=formText.isEmpty()?"/"+name.getName():shorten(formText,36); out.add(new Occurrence("FORM:"+hash,Kind.FORM_XOBJECT,pageIndex,operandStart,i+1,alpha,0f,0f,"반복 Form XObject 후보: "+shown,"Form /"+name.getName()+" · 원본 벡터/텍스트 객체")); } } catch(Exception ignored){} }
            else if("BDC".equals(op) || "BMC".equals(op)) { MarkInfo info=inspectMarkedContent(args,resources); markBlocks.push(new MarkBlock(operandStart,info.isWatermark,info.name)); }
            else if("EMC".equals(op)) { if(!markBlocks.isEmpty()){ MarkBlock m=markBlocks.pop(); if(m.watermark){ String key="TAG:"+hashTokens(tokens,m.start,i+1); Kind kind=(m.name!=null&&m.name.startsWith("OCG:"))?Kind.OPTIONAL_CONTENT:Kind.TAGGED_WATERMARK; String n=m.name==null?"Watermark":m.name.replace("OCG:",""); out.add(new Occurrence(key,kind,pageIndex,m.start,i+1,alpha,0f,0f,kind==Kind.TAGGED_WATERMARK?"확실한 Watermark marked-content":"Optional-content 후보: "+n,kind==Kind.TAGGED_WATERMARK?"/Artifact /Subtype /Watermark":"OCG/Layer: "+n)); } } }
            else if(isVisibleOperator(op)) markVisible(qBlocks);
            operandStart=i+1;
        }
        return out;
    }

    private static MarkInfo inspectMarkedContent(List<Object> args,PDResources resources){ MarkInfo info=new MarkInfo(); if(args.isEmpty())return info; COSName tag=args.get(0) instanceof COSName?(COSName)args.get(0):null;
        if(tag!=null && "Watermark".equalsIgnoreCase(tag.getName())){info.isWatermark=true;info.name="Watermark";return info;}
        if(tag!=null && "Artifact".equalsIgnoreCase(tag.getName()) && args.size()>=2){ COSDictionary dict=propertyDictionary(args.get(1),resources); if(dict!=null){COSName subtype=dict.getCOSName(COSName.SUBTYPE); if(subtype!=null&&"Watermark".equalsIgnoreCase(subtype.getName())){info.isWatermark=true;info.name="Watermark";return info;}}}
        if(tag!=null && "OC".equalsIgnoreCase(tag.getName()) && args.size()>=2){ Object prop=args.get(1); try{PDPropertyList list=null; if(prop instanceof COSName)list=resources.getProperties((COSName)prop); if(list instanceof PDOptionalContentGroup){String name=((PDOptionalContentGroup)list).getName(); if(name!=null&&!name.trim().isEmpty()){String low=name.toLowerCase(Locale.ROOT); boolean obvious=low.contains("watermark")||low.contains("confidential")||low.contains("draft")||low.contains("sample")||low.contains("wm"); if(obvious){info.isWatermark=true;info.name="OCG:"+name;}}}}catch(Exception ignored){}}
        return info; }
    private static COSDictionary propertyDictionary(Object arg,PDResources resources){ if(arg instanceof COSDictionary)return (COSDictionary)arg; if(arg instanceof COSName)try{PDPropertyList p=resources.getProperties((COSName)arg); if(p!=null&&p.getCOSObject() instanceof COSDictionary)return (COSDictionary)p.getCOSObject();}catch(Exception ignored){} return null; }
    private static COSName firstName(List<Object> args){for(int i=args.size()-1;i>=0;i--)if(args.get(i) instanceof COSName)return (COSName)args.get(i);return null;}
    private static void appendText(List<Object> args,StringBuilder sb){for(Object a:args)appendTextObject(a,sb);}
    private static void appendTextObject(Object o,StringBuilder sb){if(o instanceof COSString){String s=((COSString)o).getString();if(s!=null)sb.append(s);}else if(o instanceof COSArray){COSArray a=(COSArray)o;for(int i=0;i<a.size();i++){COSBase b=a.getObject(i);if(b instanceof COSString){String s=((COSString)b).getString();if(s!=null)sb.append(s);}}}}
    private static boolean isTextShow(String op){return "Tj".equals(op)||"TJ".equals(op)||"'".equals(op)||"\"".equals(op);}
    private static boolean isVisibleOperator(String op){return isTextShow(op)||"Do".equals(op)||"S".equals(op)||"s".equals(op)||"f".equals(op)||"F".equals(op)||"f*".equals(op)||"B".equals(op)||"B*".equals(op)||"b".equals(op)||"b*".equals(op)||"sh".equals(op)||"BI".equals(op);}
    private static void markVisible(Deque<QBlock> blocks){for(QBlock b:blocks)b.visible=true;}
    private static String normalizeText(String s){if(s==null)return "";String t=s.replace('\u0000',' ').replaceAll("\\s+"," ").trim();if(t.length()>160)t=t.substring(0,160);return t;}
    private static String shorten(String s,int n){if(s==null)return "";String t=s.replaceAll("\\s+"," ").trim();return t.length()<=n?t:t.substring(0,n)+"…";}
    private static String extractRawText(PDFormXObject form){try{PDFStreamParser p=new PDFStreamParser(form);p.parse();StringBuilder sb=new StringBuilder();List<Object> tokens=p.getTokens();int start=0;for(int i=0;i<tokens.size();i++){Object t=tokens.get(i);if(t instanceof Operator){String op=((Operator)t).getName();if(isTextShow(op))appendText(tokens.subList(start,i),sb);start=i+1;}}return normalizeText(sb.toString());}catch(Exception e){return "";}}
    private static String hashTokens(List<Object> tokens,int start,int end)throws IOException{ByteArrayOutputStream bos=new ByteArrayOutputStream();new ContentStreamWriter(bos).writeTokens(new ArrayList<Object>(tokens.subList(start,end)));return sha256(bos.toByteArray());}
    private static byte[] readAll(InputStream in)throws IOException{try{ByteArrayOutputStream out=new ByteArrayOutputStream();byte[] buf=new byte[8192];int n;while((n=in.read(buf))>=0)out.write(buf,0,n);return out.toByteArray();}finally{in.close();}}
    private static String sha256(byte[] data){try{MessageDigest md=MessageDigest.getInstance("SHA-256");byte[] d=md.digest(data);StringBuilder sb=new StringBuilder();for(byte b:d)sb.append(String.format(Locale.US,"%02x",b&0xff));return sb.toString();}catch(Exception e){return Integer.toHexString(java.util.Arrays.hashCode(data));}}
}

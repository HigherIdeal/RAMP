package com.higherideal.pdfcleaner;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentResolver;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.database.Cursor;
import android.graphics.Typeface;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import com.tom_roush.pdfbox.android.PDFBoxResourceLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int REQ_OPEN = 1001;
    private static final int REQ_SAVE = 1002;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private Uri inputUri;
    private File cachedInput;
    private List<PdfObjectCleaner.Candidate> candidates = new ArrayList<>();
    private final Set<String> selectedKeys = new HashSet<>();

    private TextView status;
    private Button analyzeButton;
    private Button chooseCandidateButton;
    private Button saveButton;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        PDFBoxResourceLoader.init(getApplicationContext());
        buildUi();
    }

    private void buildUi() {
        int pad = dp(20);
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("PDF Watermark Cleaner v2.1");
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        root.addView(title);

        TextView desc = new TextView(this);
        desc.setText("객체 기반 제거: PDF를 이미지로 변환하지 않습니다.\n텍스트 선택·검색, 벡터, 원본 이미지 품질을 유지한 채 선택한 워터마크 객체만 content stream에서 제거합니다.");
        desc.setTextSize(15);
        desc.setPadding(0, dp(12), 0, dp(18));
        root.addView(desc);

        Button open = new Button(this);
        open.setText("1. PDF 선택");
        open.setOnClickListener(v -> openPdf());
        root.addView(open, fullWidth());

        analyzeButton = new Button(this);
        analyzeButton.setText("2. 워터마크 객체 분석");
        analyzeButton.setEnabled(false);
        analyzeButton.setOnClickListener(v -> analyze());
        root.addView(analyzeButton, fullWidth());

        chooseCandidateButton = new Button(this);
        chooseCandidateButton.setText("3. 제거할 후보 선택");
        chooseCandidateButton.setEnabled(false);
        chooseCandidateButton.setOnClickListener(v -> showCandidates());
        root.addView(chooseCandidateButton, fullWidth());

        saveButton = new Button(this);
        saveButton.setText("4. 객체 제거 후 새 PDF 저장");
        saveButton.setEnabled(false);
        saveButton.setOnClickListener(v -> requestSave());
        root.addView(saveButton, fullWidth());

        status = new TextView(this);
        status.setText("PDF를 선택하세요.");
        status.setTextSize(14);
        status.setGravity(Gravity.START);
        status.setPadding(0, dp(18), 0, 0);
        root.addView(status);

        TextView note = new TextView(this);
        note.setText("주의: 디지털 서명이 있는 PDF는 내용을 수정하면 서명이 무효화됩니다. DRM/암호화 문서는 처리되지 않을 수 있습니다. 자동 선택은 보수적으로 동작하므로 후보 목록을 확인하세요.");
        note.setTextSize(12);
        note.setPadding(0, dp(24), 0, dp(8));
        root.addView(note);

        setContentView(scroll);
    }

    private LinearLayout.LayoutParams fullWidth() {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        p.setMargins(0, dp(4), 0, dp(4));
        return p;
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }

    private void openPdf() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);

        // Some Android/Samsung document providers save a perfectly valid PDF with a
        // generic MIME type (e.g. application/octet-stream). Restricting the picker
        // to application/pdf can therefore make that PDF disappear from the next open.
        // Use a broader picker filter and verify the actual PDF signature after selection.
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{
                "application/pdf",
                "application/x-pdf",
                "application/acrobat",
                "text/pdf",
                "text/x-pdf",
                "application/octet-stream"
        });
        i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(i, REQ_OPEN);
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        if (requestCode == REQ_OPEN) {
            Uri selected = data.getData();
            if (!looksLikePdf(selected)) {
                status.setText("선택한 파일이 PDF 형식이 아닙니다.");
                return;
            }
            inputUri = selected;
            try {
                getContentResolver().takePersistableUriPermission(inputUri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
            } catch (Exception ignored) {}
            candidates.clear();
            selectedKeys.clear();
            analyzeButton.setEnabled(true);
            chooseCandidateButton.setEnabled(false);
            saveButton.setEnabled(false);
            status.setText("선택됨: " + displayName(inputUri) + "\n'워터마크 객체 분석'을 누르세요.");
        } else if (requestCode == REQ_SAVE) {
            final Uri outUri = data.getData();
            try {
                getContentResolver().takePersistableUriPermission(outUri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            } catch (Exception ignored) {}
            performSave(outUri);
        }
    }

    private void analyze() {
        if (inputUri == null) return;
        setBusy(true, "PDF 구조를 분석 중입니다…\n페이지를 래스터화하지 않고 content stream / XObject / transparency / marked-content를 검사합니다.");
        worker.execute(() -> {
            try {
                cachedInput = copyUriToCache(inputUri, "input.pdf");
                List<PdfObjectCleaner.Candidate> found = PdfObjectCleaner.analyze(cachedInput);
                runOnUiThread(() -> {
                    candidates = found;
                    selectedKeys.clear();
                    for (PdfObjectCleaner.Candidate c : candidates) if (c.defaultSelected) selectedKeys.add(c.key);
                    setBusy(false, found.isEmpty()
                            ? "자동 분석에서 워터마크 후보를 찾지 못했습니다.\n이 경우 워터마크가 본문과 완전히 합쳐졌거나 특수한 인코딩/구조일 수 있습니다."
                            : "분석 완료: 후보 " + found.size() + "개.\n후보 선택에서 실제 워터마크만 체크하세요.");
                    chooseCandidateButton.setEnabled(!found.isEmpty());
                    saveButton.setEnabled(!selectedKeys.isEmpty());
                    if (!found.isEmpty()) showCandidates();
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    setBusy(false, "분석 실패: " + cleanMessage(e));
                    chooseCandidateButton.setEnabled(false);
                    saveButton.setEnabled(false);
                });
            }
        });
    }

    private void showCandidates() {
        if (candidates.isEmpty()) return;
        String[] labels = new String[candidates.size()];
        boolean[] checked = new boolean[candidates.size()];
        for (int i = 0; i < candidates.size(); i++) {
            labels[i] = candidates.get(i).displayText();
            checked[i] = selectedKeys.contains(candidates.get(i).key);
        }
        final Set<String> temp = new HashSet<>(selectedKeys);
        new AlertDialog.Builder(this)
                .setTitle("제거할 객체 선택")
                .setMultiChoiceItems(labels, checked, (d, which, isChecked) -> {
                    String key = candidates.get(which).key;
                    if (isChecked) temp.add(key); else temp.remove(key);
                })
                .setPositiveButton("적용", (d, w) -> {
                    selectedKeys.clear();
                    selectedKeys.addAll(temp);
                    saveButton.setEnabled(!selectedKeys.isEmpty());
                    status.setText("선택된 제거 후보: " + selectedKeys.size() + "개\n원본 PDF는 변경하지 않고 새 파일로 저장합니다.");
                })
                .setNegativeButton("취소", null)
                .show();
    }

    private void requestSave() {
        if (selectedKeys.isEmpty()) return;
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("application/pdf");
        String name = displayName(inputUri);
        if (name.toLowerCase().endsWith(".pdf")) name = name.substring(0, name.length() - 4);
        i.putExtra(Intent.EXTRA_TITLE, name + "_clean_object.pdf");
        i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION |
                Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(i, REQ_SAVE);
    }

    private void performSave(Uri outUri) {
        setBusy(true, "선택한 PDF 객체를 제거하고 새 PDF를 저장 중입니다…");
        worker.execute(() -> {
            File tempOut = new File(getCacheDir(), "clean_object.pdf");
            try {
                if (cachedInput == null || !cachedInput.exists()) cachedInput = copyUriToCache(inputUri, "input.pdf");
                int removed = PdfObjectCleaner.remove(cachedInput, tempOut, selectedKeys);
                try (InputStream in = new FileInputStream(tempOut);
                     OutputStream out = getContentResolver().openOutputStream(outUri, "w")) {
                    if (out == null) throw new IllegalStateException("출력 파일을 열 수 없습니다.");
                    byte[] buf = new byte[32768];
                    int n;
                    while ((n = in.read(buf)) >= 0) out.write(buf, 0, n);
                    out.flush();
                }
                runOnUiThread(() -> setBusy(false,
                        "저장 완료. 제거된 객체/블록: " + removed + "개\n텍스트·벡터·이미지는 래스터화하지 않았습니다."));
            } catch (Exception e) {
                runOnUiThread(() -> setBusy(false, "저장 실패: " + cleanMessage(e)));
            } finally {
                //noinspection ResultOfMethodCallIgnored
                tempOut.delete();
            }
        });
    }

    private boolean looksLikePdf(Uri uri) {
        if (uri == null) return false;
        try (InputStream in = getContentResolver().openInputStream(uri)) {
            if (in == null) return false;
            byte[] head = new byte[1024];
            int n = in.read(head);
            if (n <= 0) return false;
            byte[] sig = new byte[]{'%', 'P', 'D', 'F', '-'};
            for (int i = 0; i <= n - sig.length; i++) {
                boolean match = true;
                for (int j = 0; j < sig.length; j++) {
                    if (head[i + j] != sig[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) return true;
            }
        } catch (Exception ignored) {}
        return false;
    }

    private File copyUriToCache(Uri uri, String name) throws Exception {
        File f = new File(getCacheDir(), name);
        ContentResolver cr = getContentResolver();
        try (InputStream in = cr.openInputStream(uri); OutputStream out = new FileOutputStream(f)) {
            if (in == null) throw new IllegalStateException("입력 PDF를 열 수 없습니다.");
            byte[] buf = new byte[32768];
            int n;
            while ((n = in.read(buf)) >= 0) out.write(buf, 0, n);
        }
        return f;
    }

    private String displayName(Uri uri) {
        if (uri == null) return "document.pdf";
        Cursor c = null;
        try {
            c = getContentResolver().query(uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null);
            if (c != null && c.moveToFirst()) {
                int idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (idx >= 0) return c.getString(idx);
            }
        } catch (Exception ignored) {
        } finally {
            if (c != null) c.close();
        }
        String p = uri.getLastPathSegment();
        return p == null ? "document.pdf" : p;
    }

    private void setBusy(boolean busy, String msg) {
        analyzeButton.setEnabled(!busy && inputUri != null);
        chooseCandidateButton.setEnabled(!busy && !candidates.isEmpty());
        saveButton.setEnabled(!busy && !selectedKeys.isEmpty());
        status.setText(msg);
    }

    private String cleanMessage(Throwable e) {
        String m = e.getMessage();
        if (m == null || m.trim().isEmpty()) m = e.getClass().getSimpleName();
        if (m.length() > 220) m = m.substring(0, 220);
        return m;
    }

    @Override protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }
}

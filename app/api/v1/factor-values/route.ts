import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';

export const runtime = 'nodejs';

const FACTOR_VALUES_DIR = path.join(process.cwd(), 'public', 'data', 'factor_values');

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url);
    const fileParam = searchParams.get('file');
    if (!fileParam) {
        return NextResponse.json({ error: 'Missing file parameter' }, { status: 400 });
    }

    const filename = path.basename(fileParam);
    const filePath = path.join(FACTOR_VALUES_DIR, filename);

    if (!fs.existsSync(filePath)) {
        return NextResponse.json({ error: 'Factor values file not found' }, { status: 404 });
    }

    const nodeStream = fs.createReadStream(filePath);
    const webStream = Readable.toWeb(nodeStream) as ReadableStream;
    const headers = new Headers();
    headers.set('Content-Type', 'text/csv; charset=utf-8');
    headers.set('Content-Disposition', `attachment; filename="${filename}"`);

    return new NextResponse(webStream, { headers });
}

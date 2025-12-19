import fs from 'fs';
import path from 'path';
import zlib from 'zlib';
import Papa from 'papaparse';

export interface StrategyRequest {
    pool_id?: string;
    factor_ids?: number[] | null;
    train_start?: string;
    train_end?: string;
    test_start?: string;
    test_end?: string;
    target_horizons?: number[];
    target_horizon?: number;
    targetHorizons?: number[];
    strategy?: {
        type?: string;
        top_pct?: number;
        rebalance_days?: number;
    };
    factor_signals?: number[][];
    factor_signal_path?: string;
}

type Matrix = {
    rows: number;
    cols: number;
    data: Float32Array;
};

type NpyArray = {
    dtype: string;
    shape: number[];
    data: Float32Array | Float64Array | Int32Array | Uint8Array | string[];
};

type FactorTensor = {
    dates: string[];
    tickers: string[];
    factorKeys: string[];
    rows: number;
    cols: number;
    factors: number;
    data: Float32Array;
};

const BACKTRACK_DAYS = 100;
const FUTURE_DAYS = 30;
const QLIB_DATA_DIR = path.join(process.cwd(), 'data', '1555_qlib');
const FACTOR_VALUES_DIR = path.join(process.cwd(), 'public', 'data', 'factor_values');

function parseDate(dateStr: string): number {
    return new Date(`${dateStr}T00:00:00Z`).getTime();
}

function createMatrix(rows: number, cols: number, fill: number): Matrix {
    const data = new Float32Array(rows * cols);
    data.fill(fill);
    return { rows, cols, data };
}

function matrixFromJson(values: number[][]): Matrix {
    const rows = values.length;
    const cols = rows > 0 ? values[0].length : 0;
    const data = new Float32Array(rows * cols);
    for (let r = 0; r < rows; r += 1) {
        const row = values[r] || [];
        for (let c = 0; c < cols; c += 1) {
            const raw = row[c];
            const num = typeof raw === 'number' ? raw : Number(raw);
            data[r * cols + c] = Number.isFinite(num) ? num : Number.NaN;
        }
    }
    return { rows, cols, data };
}

function sliceMatrixRowsFromEnd(matrix: Matrix, length: number): Matrix {
    if (length >= matrix.rows) return matrix;
    const start = (matrix.rows - length) * matrix.cols;
    const sliced = matrix.data.subarray(start);
    const data = new Float32Array(sliced.length);
    data.set(sliced);
    return { rows: length, cols: matrix.cols, data };
}

function parseFactorId(key: string): number | null {
    if (!key.startsWith('factor_')) return null;
    const id = Number(key.replace('factor_', ''));
    return Number.isFinite(id) ? id : null;
}

function isPerStockCsv(filePath: string): boolean {
    if (!fs.existsSync(filePath)) return false;
    const firstLine = fs.readFileSync(filePath, 'utf-8').split(/\r?\n/)[0] || '';
    return /(^|,)(ticker|instrument|symbol)(,|$)/i.test(firstLine);
}

function loadFactorValuesMatrix(factorIds?: number[] | null): { dates: string[]; matrix: Matrix; factorKeys: string[] } | null {
    if (!fs.existsSync(FACTOR_VALUES_DIR)) return null;
    const csvFiles = fs.readdirSync(FACTOR_VALUES_DIR).filter((f) => f.endsWith('.csv')).sort();
    if (csvFiles.length === 0) return null;

    const merged = new Map<string, Record<string, number>>();
    const factorKeySet = new Set<string>();

    for (const file of csvFiles) {
        const csvPath = path.join(FACTOR_VALUES_DIR, file);
        if (isPerStockCsv(csvPath)) {
            continue;
        }
        const text = fs.readFileSync(csvPath, 'utf-8');
        const parsed = Papa.parse<Record<string, unknown>>(text, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
        }).data;
        for (const row of parsed) {
            const rawDate = row.date ?? row.Date;
            if (!rawDate) continue;
            const date = String(rawDate);
            const base = merged.get(date) || { date } as Record<string, number>;
            Object.entries(row).forEach(([key, value]) => {
                if (key === 'date' || key === 'Date' || !key.startsWith('factor_')) return;
                const num = typeof value === 'number' ? value : Number(value);
                base[key] = Number.isFinite(num) ? num : Number.NaN;
                factorKeySet.add(key);
            });
            merged.set(date, base);
        }
    }

    const dates = Array.from(merged.keys()).sort((a, b) => a.localeCompare(b));
    let factorKeys: string[] = [];
    if (factorIds && factorIds.length > 0) {
        factorKeys = factorIds.map((id) => `factor_${id}`);
        const missing = factorKeys.filter((key) => !factorKeySet.has(key));
        if (missing.length === factorKeys.length) {
            throw new Error('No factor values found for requested factor IDs.');
        }
        if (missing.length > 0) {
            console.warn(`[Strategy TS] Missing factor values for: ${missing.join(', ')}`);
            factorKeys = factorKeys.filter((key) => factorKeySet.has(key));
        }
    } else {
        factorKeys = Array.from(factorKeySet.values()).sort((a, b) => {
            const aId = parseFactorId(a) ?? 0;
            const bId = parseFactorId(b) ?? 0;
            return aId - bId;
        });
    }

    const rows = dates.length;
    const cols = factorKeys.length;
    if (rows === 0 || cols === 0) return null;
    const matrix = createMatrix(rows, cols, Number.NaN);
    for (let r = 0; r < rows; r += 1) {
        const row = merged.get(dates[r]);
        if (!row) continue;
        for (let c = 0; c < cols; c += 1) {
            const value = row[factorKeys[c]];
            if (typeof value === 'number' && Number.isFinite(value)) {
                matrix.data[r * cols + c] = value;
            }
        }
    }

    return { dates, matrix, factorKeys };
}

function loadPerStockFactorValues(factorIds?: number[] | null): FactorTensor | null {
    if (!fs.existsSync(FACTOR_VALUES_DIR)) return null;
    const csvFiles = fs.readdirSync(FACTOR_VALUES_DIR).filter((f) => f.endsWith('.csv')).sort();
    if (csvFiles.length === 0) return null;

    const rows: Array<{ date: string; ticker: string; values: Record<string, number> }> = [];
    const dateSet = new Set<string>();
    const tickerSet = new Set<string>();
    const factorKeySet = new Set<string>();

    for (const file of csvFiles) {
        const csvPath = path.join(FACTOR_VALUES_DIR, file);
        if (!isPerStockCsv(csvPath)) {
            continue;
        }
        const text = fs.readFileSync(csvPath, 'utf-8');
        const parsed = Papa.parse<Record<string, unknown>>(text, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
        }).data;
        for (const row of parsed) {
            const rawDate = row.date ?? row.Date;
            const rawTicker = row.ticker ?? row.Ticker ?? row.instrument ?? row.symbol ?? row.Symbol;
            if (!rawDate || !rawTicker) continue;
            const date = String(rawDate);
            const ticker = String(rawTicker);
            const values: Record<string, number> = {};
            Object.entries(row).forEach(([key, value]) => {
                if (!key.startsWith('factor_')) return;
                const num = typeof value === 'number' ? value : Number(value);
                if (Number.isFinite(num)) {
                    values[key] = num;
                } else {
                    values[key] = Number.NaN;
                }
                factorKeySet.add(key);
            });
            if (Object.keys(values).length === 0) continue;
            rows.push({ date, ticker, values });
            dateSet.add(date);
            tickerSet.add(ticker);
        }
    }

    if (rows.length === 0) return null;
    const dates = Array.from(dateSet.values()).sort((a, b) => a.localeCompare(b));
    const tickers = Array.from(tickerSet.values()).sort((a, b) => a.localeCompare(b));

    let factorKeys: string[] = [];
    if (factorIds && factorIds.length > 0) {
        factorKeys = factorIds.map((id) => `factor_${id}`);
        const missing = factorKeys.filter((key) => !factorKeySet.has(key));
        if (missing.length === factorKeys.length) {
            throw new Error('No per-stock factor values found for requested factor IDs.');
        }
        if (missing.length > 0) {
            console.warn(`[Strategy TS] Missing per-stock factor values for: ${missing.join(', ')}`);
            factorKeys = factorKeys.filter((key) => factorKeySet.has(key));
        }
    } else {
        factorKeys = Array.from(factorKeySet.values()).sort((a, b) => {
            const aId = parseFactorId(a) ?? 0;
            const bId = parseFactorId(b) ?? 0;
            return aId - bId;
        });
    }

    const rowsCount = dates.length;
    const colsCount = tickers.length;
    const factorsCount = factorKeys.length;
    if (rowsCount === 0 || colsCount === 0 || factorsCount === 0) return null;

    const data = new Float32Array(rowsCount * colsCount * factorsCount);
    data.fill(Number.NaN);

    const dateIndex = new Map(dates.map((d, i) => [d, i]));
    const tickerIndex = new Map(tickers.map((t, i) => [t, i]));
    const factorIndex = new Map(factorKeys.map((k, i) => [k, i]));

    for (const row of rows) {
        const tIdx = dateIndex.get(row.date);
        const sIdx = tickerIndex.get(row.ticker);
        if (tIdx === undefined || sIdx === undefined) continue;
        for (const [key, value] of Object.entries(row.values)) {
            const fIdx = factorIndex.get(key);
            if (fIdx === undefined) continue;
            data[(tIdx * colsCount + sIdx) * factorsCount + fIdx] = value;
        }
    }

    return {
        dates,
        tickers,
        factorKeys,
        rows: rowsCount,
        cols: colsCount,
        factors: factorsCount,
        data
    };
}

function computeCombinedSignalFromFactorTensor(tensor: FactorTensor): Matrix {
    const signals = createMatrix(tensor.rows, tensor.cols, 0);
    const means = new Float32Array(tensor.factors);
    const stds = new Float32Array(tensor.factors);
    for (let t = 0; t < tensor.rows; t += 1) {
        for (let f = 0; f < tensor.factors; f += 1) {
            let sum = 0;
            let sumSq = 0;
            let count = 0;
            for (let n = 0; n < tensor.cols; n += 1) {
                const v = tensor.data[(t * tensor.cols + n) * tensor.factors + f];
                if (!Number.isFinite(v)) continue;
                sum += v;
                sumSq += v * v;
                count += 1;
            }
            if (count < 2) {
                means[f] = 0;
                stds[f] = 0;
            } else {
                const mean = sum / count;
                const variance = Math.max(sumSq / count - mean * mean, 0);
                means[f] = mean;
                stds[f] = Math.sqrt(variance);
            }
        }

        for (let n = 0; n < tensor.cols; n += 1) {
            let sumZ = 0;
            let countZ = 0;
            for (let f = 0; f < tensor.factors; f += 1) {
                const v = tensor.data[(t * tensor.cols + n) * tensor.factors + f];
                if (!Number.isFinite(v)) continue;
                const std = stds[f];
                const mean = means[f];
                const z = std > 1e-9 ? (v - mean) / (std + 1e-9) : 0;
                sumZ += z;
                countZ += 1;
            }
            signals.data[t * tensor.cols + n] = countZ > 0 ? sumZ / countZ : 0;
        }
    }
    return signals;
}

function sliceMatrixByRowIndices(matrix: Matrix, rowIndices: number[]): Matrix {
    const rows = rowIndices.length;
    const cols = matrix.cols;
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < rowIndices.length; i += 1) {
        const srcOffset = rowIndices[i] * cols;
        const destOffset = i * cols;
        data.set(matrix.data.subarray(srcOffset, srcOffset + cols), destOffset);
    }
    return { rows, cols, data };
}

function alignByDates(
    factorDates: string[],
    signals: Matrix,
    returnDates: string[],
    returns: Matrix
): { dates: string[]; signals: Matrix; returns: Matrix } {
    const returnIndex = new Map(returnDates.map((d, i) => [d, i]));
    const keepSignal: number[] = [];
    const keepReturn: number[] = [];
    for (let i = 0; i < factorDates.length; i += 1) {
        const rIdx = returnIndex.get(factorDates[i]);
        if (rIdx !== undefined) {
            keepSignal.push(i);
            keepReturn.push(rIdx);
        }
    }
    const alignedSignals = sliceMatrixByRowIndices(signals, keepSignal);
    const alignedReturns = sliceMatrixByRowIndices(returns, keepReturn);
    const dates = keepSignal.map((i) => factorDates[i]);
    return { dates, signals: alignedSignals, returns: alignedReturns };
}

function parseZipEntries(buffer: Buffer): Array<{ name: string; compression: number; compressedSize: number; localOffset: number }> {
    const signature = 0x06054b50;
    const maxComment = 0xffff;
    let eocdOffset = -1;
    const startSearch = Math.max(0, buffer.length - 22 - maxComment);
    for (let i = buffer.length - 22; i >= startSearch; i -= 1) {
        if (buffer.readUInt32LE(i) === signature) {
            eocdOffset = i;
            break;
        }
    }
    if (eocdOffset === -1) {
        throw new Error('Invalid npz archive (EOCD missing)');
    }

    const cdSize = buffer.readUInt32LE(eocdOffset + 12);
    const cdOffset = buffer.readUInt32LE(eocdOffset + 16);
    const entries: Array<{ name: string; compression: number; compressedSize: number; localOffset: number }> = [];
    let offset = cdOffset;
    const cdEnd = cdOffset + cdSize;

    while (offset < cdEnd) {
        const sig = buffer.readUInt32LE(offset);
        if (sig !== 0x02014b50) {
            break;
        }
        const compression = buffer.readUInt16LE(offset + 10);
        const compressedSize = buffer.readUInt32LE(offset + 20);
        const nameLen = buffer.readUInt16LE(offset + 28);
        const extraLen = buffer.readUInt16LE(offset + 30);
        const commentLen = buffer.readUInt16LE(offset + 32);
        const localOffset = buffer.readUInt32LE(offset + 42);
        const name = buffer.toString('utf8', offset + 46, offset + 46 + nameLen);
        entries.push({ name, compression, compressedSize, localOffset });
        offset += 46 + nameLen + extraLen + commentLen;
    }
    return entries;
}

function readZipEntry(buffer: Buffer, entry: { name: string; compression: number; compressedSize: number; localOffset: number }): Buffer {
    const localSig = buffer.readUInt32LE(entry.localOffset);
    if (localSig !== 0x04034b50) {
        throw new Error(`Invalid zip entry header for ${entry.name}`);
    }
    const nameLen = buffer.readUInt16LE(entry.localOffset + 26);
    const extraLen = buffer.readUInt16LE(entry.localOffset + 28);
    const dataStart = entry.localOffset + 30 + nameLen + extraLen;
    const compressed = buffer.subarray(dataStart, dataStart + entry.compressedSize);
    if (entry.compression === 0) {
        return Buffer.from(compressed);
    }
    if (entry.compression === 8) {
        return zlib.inflateRawSync(compressed);
    }
    throw new Error(`Unsupported compression method ${entry.compression} for ${entry.name}`);
}

function parseNpy(buffer: Buffer): NpyArray {
    if (buffer.length < 10) {
        throw new Error('Invalid .npy file (too small)');
    }
    const magic = buffer.toString('latin1', 0, 6);
    if (magic !== '\x93NUMPY') {
        throw new Error('Invalid .npy file (bad magic)');
    }
    const major = buffer[6];
    let headerLen = 0;
    let headerOffset = 0;
    if (major === 1) {
        headerLen = buffer.readUInt16LE(8);
        headerOffset = 10;
    } else if (major === 2 || major === 3) {
        headerLen = buffer.readUInt32LE(8);
        headerOffset = 12;
    } else {
        throw new Error(`Unsupported .npy version ${major}`);
    }
    const header = buffer.toString('latin1', headerOffset, headerOffset + headerLen);
    const descrMatch = header.match(/'descr':\s*'([^']+)'/);
    const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
    if (!descrMatch || !shapeMatch) {
        throw new Error('Invalid .npy header');
    }
    const dtype = descrMatch[1];
    const shape = shapeMatch[1]
        .split(',')
        .map((part) => part.trim())
        .filter((part) => part.length > 0)
        .map((part) => Number(part))
        .filter((num) => Number.isFinite(num));
    const size = shape.reduce((acc, val) => acc * val, 1);
    const dataOffset = headerOffset + headerLen;

    if (dtype.includes('U')) {
        const lenMatch = dtype.match(/U(\d+)/);
        const strLen = lenMatch ? Number(lenMatch[1]) : 0;
        const totalChars = size * strLen;
        const byteLength = totalChars * 4;
        const view = new DataView(buffer.buffer, buffer.byteOffset + dataOffset, byteLength);
        const out: string[] = new Array(size);
        let ptr = 0;
        for (let i = 0; i < size; i += 1) {
            let text = '';
            for (let j = 0; j < strLen; j += 1) {
                const code = view.getUint32(ptr, true);
                ptr += 4;
                if (code !== 0) {
                    text += String.fromCodePoint(code);
                }
            }
            out[i] = text;
        }
        return { dtype, shape, data: out };
    }

    if (dtype.endsWith('f4')) {
        const byteLength = size * 4;
        const sliced = buffer.subarray(dataOffset, dataOffset + byteLength);
        const view = sliced.byteOffset % 4 === 0
            ? new Float32Array(sliced.buffer, sliced.byteOffset, size)
            : (() => {
                const copy = Buffer.from(sliced);
                return new Float32Array(copy.buffer, copy.byteOffset, size);
            })();
        return { dtype, shape, data: view };
    }
    if (dtype.endsWith('f8')) {
        const byteLength = size * 8;
        const sliced = buffer.subarray(dataOffset, dataOffset + byteLength);
        const view = sliced.byteOffset % 8 === 0
            ? new Float64Array(sliced.buffer, sliced.byteOffset, size)
            : (() => {
                const copy = Buffer.from(sliced);
                return new Float64Array(copy.buffer, copy.byteOffset, size);
            })();
        return { dtype, shape, data: view };
    }
    if (dtype.endsWith('i4')) {
        const byteLength = size * 4;
        const sliced = buffer.subarray(dataOffset, dataOffset + byteLength);
        const view = sliced.byteOffset % 4 === 0
            ? new Int32Array(sliced.buffer, sliced.byteOffset, size)
            : (() => {
                const copy = Buffer.from(sliced);
                return new Int32Array(copy.buffer, copy.byteOffset, size);
            })();
        return { dtype, shape, data: view };
    }
    if (dtype.endsWith('u1')) {
        const view = new Uint8Array(buffer.subarray(dataOffset, dataOffset + size));
        return { dtype, shape, data: view };
    }
    throw new Error(`Unsupported dtype ${dtype}`);
}

function readNpz(filePath: string): Record<string, NpyArray> {
    const buffer = fs.readFileSync(filePath);
    const entries = parseZipEntries(buffer);
    const result: Record<string, NpyArray> = {};
    for (const entry of entries) {
        const name = entry.name.replace(/\.npy$/i, '');
        const raw = readZipEntry(buffer, entry);
        result[name] = parseNpy(raw);
    }
    return result;
}

function matrixFromNpy(array: NpyArray, label: string): Matrix {
    if (array.shape.length !== 2) {
        throw new Error(`Expected ${label} to be 2D, got shape ${array.shape.join('x')}`);
    }
    const [rows, cols] = array.shape;
    if (array.data instanceof Float32Array) {
        return { rows, cols, data: array.data };
    }
    if (array.data instanceof Float64Array) {
        const data = new Float32Array(array.data.length);
        for (let i = 0; i < array.data.length; i += 1) data[i] = array.data[i];
        return { rows, cols, data };
    }
    if (array.data instanceof Int32Array || array.data instanceof Uint8Array) {
        const data = new Float32Array(array.data.length);
        for (let i = 0; i < array.data.length; i += 1) data[i] = array.data[i];
        return { rows, cols, data };
    }
    throw new Error(`Unsupported ${label} dtype ${array.dtype}`);
}

function loadClosePrices(
    qlibPath: string,
    startDate: string,
    endDate: string,
    stockIds?: string[]
): { dates: string[]; stockIds: string[]; close: Matrix } {
    const calendarPath = path.join(qlibPath, 'calendars', 'day.txt');
    const calendarText = fs.readFileSync(calendarPath, 'utf-8');
    const allDates = calendarText.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const startTime = parseDate(startDate);
    const endTime = parseDate(endDate);

    let startIdx = 0;
    for (let i = 0; i < allDates.length; i += 1) {
        if (parseDate(allDates[i]) >= startTime) {
            startIdx = i;
            break;
        }
    }
    let endIdx = allDates.length;
    for (let i = 0; i < allDates.length; i += 1) {
        if (parseDate(allDates[i]) > endTime) {
            endIdx = i;
            break;
        }
    }
    startIdx = Math.max(0, startIdx - BACKTRACK_DAYS);
    endIdx = Math.min(allDates.length, endIdx + FUTURE_DAYS);

    const dates = allDates.slice(startIdx, endIdx);
    let resolvedStockIds: string[] = [];
    if (stockIds && stockIds.length > 0) {
        resolvedStockIds = stockIds.map((id) => id.toLowerCase());
    } else {
        const instrumentsPath = path.join(qlibPath, 'instruments', 'all.txt');
        const instrumentsText = fs.readFileSync(instrumentsPath, 'utf-8');
        resolvedStockIds = instrumentsText
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean)
            .map((line) => line.split('\t')[0].toLowerCase());
    }

    const nDays = dates.length;
    const nStocks = resolvedStockIds.length;
    const close = createMatrix(nDays, nStocks, Number.NaN);
    const featuresDir = path.join(qlibPath, 'features');

    for (let sIdx = 0; sIdx < nStocks; sIdx += 1) {
        const stockId = resolvedStockIds[sIdx];
        const closeFile = path.join(featuresDir, stockId, 'close.day.bin');
        if (!fs.existsSync(closeFile)) continue;
        const buffer = fs.readFileSync(closeFile);
        const size = Math.floor(buffer.byteLength / 4);
        const data = buffer.byteOffset % 4 === 0
            ? new Float32Array(buffer.buffer, buffer.byteOffset, size)
            : new Float32Array(Buffer.from(buffer).buffer);
        if (data.length >= endIdx) {
            for (let d = 0; d < nDays; d += 1) {
                close.data[d * nStocks + sIdx] = data[startIdx + d];
            }
        } else if (data.length > startIdx) {
            const available = Math.min(data.length - startIdx, nDays);
            for (let d = 0; d < available; d += 1) {
                close.data[d * nStocks + sIdx] = data[startIdx + d];
            }
        }
    }
    return { dates, stockIds: resolvedStockIds, close };
}

function computeReturns(close: Matrix): Matrix {
    const returns = createMatrix(close.rows, close.cols, 0);
    for (let r = 1; r < close.rows; r += 1) {
        const rowOffset = r * close.cols;
        const prevOffset = (r - 1) * close.cols;
        for (let c = 0; c < close.cols; c += 1) {
            const curr = close.data[rowOffset + c];
            const prev = close.data[prevOffset + c];
            if (!Number.isFinite(curr) || !Number.isFinite(prev)) {
                returns.data[rowOffset + c] = 0;
            } else {
                const denom = Math.max(prev, 1e-8);
                returns.data[rowOffset + c] = curr / denom - 1;
            }
        }
    }
    return returns;
}

function nanToZero(matrix: Matrix): Matrix {
    const data = new Float32Array(matrix.data.length);
    for (let i = 0; i < matrix.data.length; i += 1) {
        const v = matrix.data[i];
        data[i] = Number.isFinite(v) ? v : 0;
    }
    return { rows: matrix.rows, cols: matrix.cols, data };
}

function nanToZeroAndClip(matrix: Matrix, minValue: number, maxValue: number): Matrix {
    const data = new Float32Array(matrix.data.length);
    for (let i = 0; i < matrix.data.length; i += 1) {
        const v = matrix.data[i];
        if (!Number.isFinite(v)) {
            data[i] = 0;
            continue;
        }
        if (v < minValue) {
            data[i] = minValue;
        } else if (v > maxValue) {
            data[i] = maxValue;
        } else {
            data[i] = v;
        }
    }
    return { rows: matrix.rows, cols: matrix.cols, data };
}

function computeDailyIc(signals: Matrix, returns: Matrix): Float32Array {
    const ic = new Float32Array(signals.rows);
    ic.fill(Number.NaN);
    for (let t = 0; t < signals.rows; t += 1) {
        let count = 0;
        let sumS = 0;
        let sumR = 0;
        let sumSS = 0;
        let sumRR = 0;
        let sumSR = 0;
        const offset = t * signals.cols;
        for (let i = 0; i < signals.cols; i += 1) {
            const s = signals.data[offset + i];
            const r = returns.data[offset + i];
            if (!Number.isFinite(s) || !Number.isFinite(r)) continue;
            count += 1;
            sumS += s;
            sumR += r;
            sumSS += s * s;
            sumRR += r * r;
            sumSR += s * r;
        }
        if (count < 2) continue;
        const meanS = sumS / count;
        const meanR = sumR / count;
        const denom = count - 1;
        const varS = (sumSS - count * meanS * meanS) / denom;
        const varR = (sumRR - count * meanR * meanR) / denom;
        if (varS <= 0 || varR <= 0) continue;
        const cov = (sumSR - count * meanS * meanR) / denom;
        ic[t] = cov / Math.sqrt(varS * varR);
    }
    return ic;
}

function computePortfolioReturns(
    signals: Matrix,
    returns: Matrix,
    topPct: number,
    stratType: string,
    rebalanceDays: number
): { portfolio: Float32Array; benchmark: Float32Array; rebalanceIds: Int32Array } {
    const rows = signals.rows;
    const cols = signals.cols;
    const pct = Math.max(Math.min(topPct, 0.5), 0.01);
    const step = Math.max(Math.floor(rebalanceDays), 1);
    const portfolio = new Float32Array(rows);
    const benchmark = new Float32Array(rows);
    const rebalanceIds = new Int32Array(rows);

    const weights = new Float32Array(cols);
    let rebalanceIdx = 0;
    for (let start = 0; start < rows; start += step) {
        const end = Math.min(start + step, rows);
        const rowOffset = start * cols;
        const indices = Array.from({ length: cols }, (_, i) => i);
        indices.sort((a, b) => {
            const diff = signals.data[rowOffset + a] - signals.data[rowOffset + b];
            return diff === 0 ? a - b : diff;
        });
        weights.fill(0);
        const longThreshold = cols * (1 - pct);
        const shortThreshold = cols * pct;
        let longCount = 0;
        let shortCount = 0;
        const ranks = new Float32Array(cols);
        for (let rank = 0; rank < cols; rank += 1) {
            ranks[indices[rank]] = rank;
        }
        for (let i = 0; i < cols; i += 1) {
            const rank = ranks[i];
            if (rank >= longThreshold) longCount += 1;
            if (rank < shortThreshold) shortCount += 1;
        }
        longCount = Math.max(longCount, 1);
        shortCount = Math.max(shortCount, 1);
        for (let i = 0; i < cols; i += 1) {
            const rank = ranks[i];
            if (rank >= longThreshold) weights[i] = 1 / longCount;
            if (stratType === 'long_short' && rank < shortThreshold) weights[i] = -1 / shortCount;
            if (stratType === 'equal_weight') weights[i] = 1 / cols;
        }

        const cum = new Float32Array(cols);
        cum.fill(1);
        let prevNav = 1;
        for (let t = start; t < end; t += 1) {
            let navContribution = 0;
            const offset = t * cols;
            for (let i = 0; i < cols; i += 1) {
                const r = returns.data[offset + i];
                const next = cum[i] * (1 + r);
                cum[i] = next;
                navContribution += weights[i] * (next - 1);
            }
            const nav = 1 + navContribution;
            portfolio[t] = nav / Math.max(prevNav, 1e-8) - 1;
            rebalanceIds[t] = rebalanceIdx;
            prevNav = nav;
        }
        rebalanceIdx += 1;
    }

    const validMask = new Array(cols).fill(true);
    let validCount = cols;
    for (let i = 0; i < cols; i += 1) {
        const v = returns.data[i];
        if (!Number.isFinite(v)) {
            validMask[i] = false;
            validCount -= 1;
        }
    }
    validCount = Math.max(validCount, 1);
    const benchWeights = new Float32Array(cols);
    for (let i = 0; i < cols; i += 1) {
        if (validMask[i]) benchWeights[i] = 1 / validCount;
    }
    const benchCum = new Float32Array(cols);
    benchCum.fill(1);
    let prevBench = 1;
    for (let t = 0; t < rows; t += 1) {
        let nav = 0;
        const offset = t * cols;
        for (let i = 0; i < cols; i += 1) {
            const r = returns.data[offset + i];
            benchCum[i] = benchCum[i] * (1 + r);
            nav += benchWeights[i] * benchCum[i];
        }
        benchmark[t] = nav / Math.max(prevBench, 1e-8) - 1;
        prevBench = nav;
    }

    return { portfolio, benchmark, rebalanceIds };
}

function mean(values: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) sum += values[i];
    return values.length > 0 ? sum / values.length : 0;
}

function std(values: Float32Array): number {
    if (values.length === 0) return 0;
    const avg = mean(values);
    let variance = 0;
    for (let i = 0; i < values.length; i += 1) {
        const diff = values[i] - avg;
        variance += diff * diff;
    }
    variance /= values.length;
    return Math.sqrt(variance);
}

function nanMean(values: Float32Array): number {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < values.length; i += 1) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        sum += v;
        count += 1;
    }
    return count > 0 ? sum / count : Number.NaN;
}

function nanStd(values: Float32Array): number {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < values.length; i += 1) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        sum += v;
        count += 1;
    }
    if (count === 0) return Number.NaN;
    const avg = sum / count;
    let variance = 0;
    for (let i = 0; i < values.length; i += 1) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        const diff = v - avg;
        variance += diff * diff;
    }
    variance /= count;
    return Math.sqrt(variance);
}

function sliceByMask(values: Float32Array, mask: boolean[]): Float32Array {
    const out = new Float32Array(mask.filter(Boolean).length);
    let idx = 0;
    for (let i = 0; i < values.length; i += 1) {
        if (mask[i]) {
            out[idx] = values[i];
            idx += 1;
        }
    }
    return out;
}

function sliceIcByMask(values: Float32Array, mask: boolean[]): Float32Array {
    return sliceByMask(values, mask);
}

function computeMetrics(
    retSlice: Float32Array,
    icSlice: Float32Array,
    period: string,
    stratType: string,
    horizonDays: number
): Record<string, number | string> | null {
    if (retSlice.length === 0) return null;
    const avgRet = mean(retSlice);
    const stdRet = std(retSlice);
    const sharpe = (avgRet / (stdRet + 1e-9)) * Math.sqrt(252);
    let nav = 1;
    let runMax = 1;
    let mdd = 0;
    for (let i = 0; i < retSlice.length; i += 1) {
        nav *= 1 + retSlice[i];
        if (nav > runMax) runMax = nav;
        const dd = (nav - runMax) / runMax;
        if (dd < mdd) mdd = dd;
    }
    const icMean = nanMean(icSlice);
    const icStd = nanStd(icSlice);
    const icir = icMean / (icStd + 1e-9);
    return {
        factor_id: -1,
        factor_name: `Strategy (${stratType})`,
        horizon_days: horizonDays,
        period,
        ic: icMean,
        ic_std: icStd,
        icir,
        ric: 0,
        sharpe,
        annual_return: avgRet * 252,
        daily_return_mean: avgRet,
        max_drawdown: mdd,
    };
}

export async function computeStrategyFallback(request: StrategyRequest): Promise<Record<string, unknown>> {
    const t0 = Date.now();
    const trainStart = request.train_start ?? '2022-01-01';
    const trainEnd = request.train_end ?? '2023-12-31';
    const testStart = request.test_start ?? '2024-01-01';
    const testEnd = request.test_end ?? '2024-12-31';

    const strategyCfg = request.strategy ?? {};
    const stratType = strategyCfg.type ?? 'long_short';
    const topPct = strategyCfg.top_pct ?? 0.2;
    const rebalanceDays = strategyCfg.rebalance_days ?? 1;

    const horizonInput = request.target_horizons ?? request.targetHorizons ?? request.target_horizon;
    let horizonDays = 1;
    if (Array.isArray(horizonInput) && horizonInput.length > 0) {
        const parsed = Number(horizonInput[0]);
        if (Number.isFinite(parsed)) horizonDays = parsed;
    } else if (typeof horizonInput === 'number' || typeof horizonInput === 'string') {
        const parsed = Number(horizonInput);
        if (Number.isFinite(parsed)) horizonDays = parsed;
    }
    horizonDays = Math.max(Math.floor(horizonDays), 1);

    let factorSignals: Matrix | null = null;
    let targetReturns: Matrix | null = null;
    let dates: string[] | null = null;
    let returnDates: string[] | null = null;
    let stockIds: string[] | null = null;
    let clipReturns = true;

    if (request.factor_signal_path) {
        try {
            const loaded = readNpz(request.factor_signal_path);
            if (loaded.signal) factorSignals = matrixFromNpy(loaded.signal, 'signal');
            if (loaded.target_returns) targetReturns = matrixFromNpy(loaded.target_returns, 'target_returns');
            if (loaded.dates && Array.isArray(loaded.dates.data)) {
                dates = loaded.dates.data as string[];
            }
        } catch (err) {
            console.error('[Strategy TS] Failed to load cached npz:', err);
        }
    }

    if (!factorSignals && request.factor_signals) {
        factorSignals = matrixFromJson(request.factor_signals);
    }
    if (!factorSignals) {
        const perStock = loadPerStockFactorValues(request.factor_ids ?? null);
        if (perStock) {
            factorSignals = computeCombinedSignalFromFactorTensor(perStock);
            dates = perStock.dates;
            stockIds = perStock.tickers;
            console.log(`[Strategy TS] Loaded per-stock factor values: ${perStock.rows}x${perStock.cols}x${perStock.factors}`);
        }
    }
    if (!factorSignals) {
        const factorValues = loadFactorValuesMatrix(request.factor_ids ?? null);
        if (factorValues) {
            factorSignals = factorValues.matrix;
            targetReturns = factorValues.matrix;
            dates = factorValues.dates;
            clipReturns = false;
            console.log(`[Strategy TS] Loaded factor values: ${factorValues.dates.length}x${factorValues.factorKeys.length}`);
        }
    }
    if (!factorSignals) {
        throw new Error('Factor signals missing; provide factor_signals or factor_values CSVs in public/data/factor_values.');
    }

    if (!targetReturns || !dates || dates.length !== factorSignals.rows) {
        const loaded = loadClosePrices(QLIB_DATA_DIR, trainStart, testEnd, stockIds ?? undefined);
        targetReturns = computeReturns(loaded.close);
        returnDates = loaded.dates;
        stockIds = loaded.stockIds;
        if (dates && returnDates && dates.length !== returnDates.length) {
            const aligned = alignByDates(dates, factorSignals, returnDates, targetReturns);
            dates = aligned.dates;
            factorSignals = aligned.signals;
            targetReturns = aligned.returns;
        } else if (!dates) {
            dates = returnDates;
        }
        if (targetReturns.rows !== factorSignals.rows) {
            const minLen = Math.min(targetReturns.rows, factorSignals.rows);
            targetReturns = sliceMatrixRowsFromEnd(targetReturns, minLen);
            factorSignals = sliceMatrixRowsFromEnd(factorSignals, minLen);
            dates = (dates || []).slice((dates || []).length - minLen);
        }
    }

    if (!targetReturns || !dates) {
        throw new Error('Missing stock returns or date metadata.');
    }
    if (factorSignals.rows !== targetReturns.rows || factorSignals.cols !== targetReturns.cols) {
        throw new Error(`Shape mismatch between factor signals (${factorSignals.rows}, ${factorSignals.cols}) and stock returns (${targetReturns.rows}, ${targetReturns.cols})`);
    }
    if (!dates || dates.length !== factorSignals.rows) {
        throw new Error('Date metadata length mismatch; regenerate cached signals.');
    }

    console.log(`[Strategy TS] signals ${factorSignals.rows}x${factorSignals.cols}, returns ${targetReturns.rows}x${targetReturns.cols}, dates ${dates.length}`);

    const dailyIc = computeDailyIc(factorSignals, targetReturns);
    const signalsClean = nanToZero(factorSignals);
    const returnsClean = clipReturns
        ? nanToZeroAndClip(targetReturns, -0.9, 0.5)
        : nanToZero(targetReturns);

    const { portfolio, benchmark, rebalanceIds } = computePortfolioReturns(
        signalsClean,
        returnsClean,
        topPct,
        stratType,
        rebalanceDays
    );

    const trainStartTime = parseDate(trainStart);
    const trainEndTime = parseDate(trainEnd);
    const testStartTime = parseDate(testStart);
    const testEndTime = parseDate(testEnd);
    const trainMask = dates.map((d) => {
        const ts = parseDate(d);
        return ts >= trainStartTime && ts <= trainEndTime;
    });
    const testMask = dates.map((d) => {
        const ts = parseDate(d);
        return ts >= testStartTime && ts <= testEndTime;
    });

    const metrics: Record<string, unknown>[] = [];
    const dailyReturns: Record<string, unknown>[] = [];

    const trainRet = sliceByMask(portfolio, trainMask);
    const trainIc = sliceIcByMask(dailyIc, trainMask);
    const trainMetrics = computeMetrics(trainRet, trainIc, 'train', stratType, horizonDays);
    if (trainMetrics) metrics.push(trainMetrics);
    for (let i = 0; i < dates.length; i += 1) {
        if (!trainMask[i]) continue;
        dailyReturns.push({
            date: dates[i],
            factor_id: -1,
            horizon_days: horizonDays,
            return: portfolio[i],
            benchmark: benchmark[i],
            rebalance_id: rebalanceIds[i],
        });
    }

    const testRet = sliceByMask(portfolio, testMask);
    const testIc = sliceIcByMask(dailyIc, testMask);
    const testMetrics = computeMetrics(testRet, testIc, 'test', stratType, horizonDays);
    if (testMetrics) metrics.push(testMetrics);
    for (let i = 0; i < dates.length; i += 1) {
        if (!testMask[i]) continue;
        dailyReturns.push({
            date: dates[i],
            factor_id: -1,
            horizon_days: horizonDays,
            return: portfolio[i],
            benchmark: benchmark[i],
            rebalance_id: rebalanceIds[i],
        });
    }

    return {
        metrics,
        daily_returns: dailyReturns,
        computation_time_ms: Date.now() - t0,
    };
}

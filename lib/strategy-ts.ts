import fs from 'fs';
import path from 'path';
import zlib from 'zlib';

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

const BACKTRACK_DAYS = 100;
const FUTURE_DAYS = 30;
const QLIB_DATA_DIR = path.join(process.cwd(), 'data', '1555_qlib');

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

function loadClosePrices(qlibPath: string, startDate: string, endDate: string): { dates: string[]; close: Matrix } {
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
    const instrumentsPath = path.join(qlibPath, 'instruments', 'all.txt');
    const instrumentsText = fs.readFileSync(instrumentsPath, 'utf-8');
    const stockIds = instrumentsText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => line.split('\t')[0].toLowerCase());

    const nDays = dates.length;
    const nStocks = stockIds.length;
    const close = createMatrix(nDays, nStocks, Number.NaN);
    const featuresDir = path.join(qlibPath, 'features');

    for (let sIdx = 0; sIdx < nStocks; sIdx += 1) {
        const stockId = stockIds[sIdx];
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
    return { dates, close };
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
        throw new Error('Factor signals missing; ensure pool_id/factor selections are provided.');
    }

    if (!targetReturns || !dates || dates.length !== factorSignals.rows) {
        const loaded = loadClosePrices(QLIB_DATA_DIR, trainStart, testEnd);
        targetReturns = computeReturns(loaded.close);
        dates = loaded.dates;
        if (targetReturns.rows !== factorSignals.rows) {
            const minLen = Math.min(targetReturns.rows, factorSignals.rows);
            targetReturns = sliceMatrixRowsFromEnd(targetReturns, minLen);
            factorSignals = sliceMatrixRowsFromEnd(factorSignals, minLen);
            dates = dates.slice(dates.length - minLen);
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
    const returnsClean = nanToZeroAndClip(targetReturns, -0.9, 0.5);

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

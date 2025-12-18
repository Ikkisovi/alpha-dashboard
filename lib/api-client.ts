
import { ComputeRequest, ComputeResponse } from './types';

export class ApiClient {
    private static baseUrl = '/api/v1';

    static async computeMetrics(request: ComputeRequest): Promise<ComputeResponse> {
        const response = await fetch(`${this.baseUrl}/compute/metrics`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(request),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.error || `HTTP error ${response.status}`);
        }

        return response.json();
    }
}

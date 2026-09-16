export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set('X-PhotoForm-Client', 'local');
  const response = await fetch(path, { ...init, headers });
  const value = await response.json();
  if (!response.ok) throw new ApiError(typeof value.detail === 'string' ? value.detail : '请求失败，请检查参数。', response.status);
  return value as T;
}

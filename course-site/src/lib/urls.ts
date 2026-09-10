export function withBase(path = '') {
  const base = import.meta.env.BASE_URL;
  return `${base}${path.replace(/^\//, '')}`;
}

export function colabUrl(repository: string, source: string, branch = 'main', notebookRoot = '') {
  const path = [notebookRoot, source]
    .filter(Boolean)
    .map((part) => part.replace(/^\/+|\/+$/g, ''))
    .join('/');
  return `https://colab.research.google.com/github/${repository.replace(/^https:\/\/github\.com\//, '').replace(/^\/+|\/+$/g, '')}/blob/${branch}/${path}`;
}

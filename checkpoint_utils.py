import os
import time
import glob
import tempfile
import torch


def save_checkpoint(path, model, optimizer=None, epoch=None, scheduler=None, extra=None):
    """Salva checkpoint in modo atomico. Salva model_state_dict sempre; aggiunge optimizer/scheduler/epoch se forniti."""
    data = {
        'model_state_dict': model.state_dict(),
        'timestamp': time.time(),
    }
    if optimizer is not None:
        data['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        try:
            data['scheduler_state_dict'] = scheduler.state_dict()
        except Exception:
            pass
    if epoch is not None:
        data['epoch'] = int(epoch)
    if extra is not None:
        data['extra'] = extra

    dirn = os.path.dirname(path)
    if dirn and not os.path.exists(dirn):
        os.makedirs(dirn, exist_ok=True)

    tmp = path + '.tmp'
    torch.save(data, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu', strict=True):
    """Carica checkpoint. Supporta file salvati come dict (con chiave 'model_state_dict') o raw state_dict.
    Se presente, carica anche optimizer/scheduler. Ritorna il dict del checkpoint.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=map_location)
    # gestisci sia dict completo che solo state_dict
    if isinstance(ck, dict) and ('model_state_dict' in ck):
        state = ck['model_state_dict']
    else:
        state = ck

    model.load_state_dict(state, strict=strict)

    if optimizer is not None and isinstance(ck, dict) and ('optimizer_state_dict' in ck):
        optimizer.load_state_dict(ck['optimizer_state_dict'])

    if scheduler is not None and isinstance(ck, dict) and ('scheduler_state_dict' in ck):
        try:
            scheduler.load_state_dict(ck['scheduler_state_dict'])
        except Exception:
            pass

    return ck


def find_latest_checkpoint(dirpath, pattern='*'):
    """Ritorna il file più recente che corrisponde a pattern (glob) nella cartella. None se non trovato."""
    if not os.path.exists(dirpath):
        return None
    files = glob.glob(os.path.join(dirpath, pattern))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


__all__ = ['save_checkpoint', 'load_checkpoint', 'find_latest_checkpoint']

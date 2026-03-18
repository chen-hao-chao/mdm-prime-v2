import torch

class BasebLayer(torch.nn.Module):
    def __init__(self, base, target_length, device='cuda') -> None:
        super().__init__()
        self.base = base
        self.target_length = target_length

        # Precompute powers for encoding and decoding
        mask = (base ** torch.arange(target_length - 1, -1, -1)).to(device)
        self.register_buffer("mask", mask)

        # Precompute lookup table for all possible integer values
        max_value = base ** target_length
        values = torch.arange(max_value).to(device)
        digits = (values.unsqueeze(-1) // mask) % base
        self.register_buffer("lookup_table", digits.long())  # [max_value, target_length]

    def forward(self, x):
        if self.target_length == 1:
            return x
        B, L = x.shape
        flat_x = x.reshape(B*L,)
        y_code = self.lookup_table[flat_x]  # [B*L, target_length]
        y = y_code.view(B, -1)  # [B, target_length * seq_len]
        return y

    def inverse(self, y):
        if self.target_length == 1:
            return y

        B, L_l = y.shape
        L = L_l // self.target_length
        y = y.view(B, L, self.target_length)
        x = (y * self.mask.to(y.dtype)).sum(dim=-1)
        return x

class BasebShufflingLayer(torch.nn.Module):
    def __init__(self, base, target_length, perm=None, random_ratio: float = 1.0) -> None:
        """
        random_ratio in [0, 1]:
          - 1.0 -> fully random permutation of all max_value codes
          - 0.5 -> only the first 50% of codes [0, ..., 0.5*max_value-1] are permuted;
                   the rest are left as identity
          - 0.0 -> identity permutation
        """
        super().__init__()
        self.base = base
        self.target_length = target_length # token granularity
        self.random_ratio = float(random_ratio)

        if target_length == 1:
            # No encoding/decoding structure needed in this trivial case
            self.register_buffer("mask", torch.tensor([], dtype=torch.long))
            self.register_buffer("lookup_table", torch.tensor([], dtype=torch.long))
            self.register_buffer("perm", torch.tensor([], dtype=torch.long))
            self.register_buffer("inv_perm", torch.tensor([], dtype=torch.long))
            return

        # Precompute powers for encoding and decoding
        # e.g. for target_length=3 and base=b: [b^2, b^1, b^0]
        mask = (base ** torch.arange(target_length - 1, -1, -1)).to('cuda')
        self.register_buffer("mask", mask)

        # Precompute lookup table for all possible integer values
        max_value = base ** target_length
        values = torch.arange(max_value, device='cuda')  # [max_value]
        digits = (values.unsqueeze(-1) // mask) % base   # [max_value, target_length]
        self.register_buffer("lookup_table", digits.long())

        # perm: maps "original index" -> "permuted index"
        if perm is not None:
            # Use user-provided permutation (assumed to be a full permutation)
            perm = perm.to(device='cuda', dtype=torch.long)
        else:
            # Create permutation according to random_ratio
            # Clamp ratio to [0, 1] to be safe
            rr = max(0.0, min(1.0, self.random_ratio))

            if rr >= 1.0:
                # Fully random permutation over all max_value codes
                perm = torch.randperm(max_value, device='cuda')
            elif rr <= 0.0:
                # Identity permutation (no randomization)
                perm = torch.arange(max_value, device='cuda')
            else:
                # Partially random:
                # only the first K indices are permuted among themselves,
                # the rest remain identity.
                K = int(max_value * rr)
                K = max(0, min(max_value, K))  # just in case

                # Start from identity
                perm = torch.arange(max_value, device='cuda')

                if K > 1:
                    rand_idx = torch.randperm(K, device='cuda')  # permute [0..K-1]
                    perm[:K] = perm[:K][rand_idx]
                # if K == 0 or 1, perm remains identity (or trivial)

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(max_value, device='cuda')

        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x):
        """
        x: [B, L] integer codes in [0, base^target_length - 1]
        Returns y: [B, L * target_length] base-b digits of permuted codes.
        """
        if self.target_length == 1:
            return x

        B, L = x.shape
        flat_x = x.reshape(B * L).long()             # [B*L]
        flat_x_perm = self.perm[flat_x]              # apply permutation: [B*L]
        y_code = self.lookup_table[flat_x_perm]      # [B*L, target_length]
        y = y_code.view(B, -1)                       # [B, L * target_length]
        return y

    def inverse(self, y):
        """
        y: [B, L * target_length] base-b digits (of permuted codes)
        Returns x: [B, L] original integer codes.
        """
        if self.target_length == 1:
            return y

        B, L_l = y.shape
        L = L_l // self.target_length

        # [B, L * target_length] -> [B*L, target_length]
        y = y.view(B * L, self.target_length)

        # Decode base-b digits back to permuted integer codes
        x_perm = (y.to(self.mask.dtype) * self.mask).sum(dim=-1).long()  # [B*L]

        # Undo permutation to get original codes
        x = self.inv_perm[x_perm]  # [B*L]

        return x.view(B, L)

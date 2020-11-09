#ifndef __UNINITIALIZED_ALLOCATOR_H__
#define __UNINITIALIZED_ALLOCATOR_H__

#include <memory>
#include <vector>

// below code is taken from https://stackoverflow.com/questions/15952412

// based on a design by Jared Hoberock
// edited (Walter) 10-May-2013, 23-Apr-2014
template <typename T, typename base_allocator = std::allocator<T>>
struct uninitialized_allocator : base_allocator {
    static_assert(std::is_same<T, typename base_allocator::value_type>::value,
                  "allocator::value_type mismatch");

    template <typename U>
    using base_t = typename std::allocator_traits<
        base_allocator>::template rebind_alloc<U>;

    // rebind to base_t<U> for all U!=T: we won't leave other types
    // uninitialized!
    template <typename U> struct rebind {
        typedef
            typename std::conditional<std::is_same<T, U>::value,
                                      uninitialized_allocator, base_t<U>>::type
                other;
    };

    // elide trivial default construction of objects of type T only
    template <typename U>
    typename std::enable_if<
        std::is_same<T, U>::value &&
        std::is_trivially_default_constructible<U>::value>::type
    construct(U *)
    {
    }

    // elide trivial default destruction of objects of type T only
    template <typename U>
    typename std::enable_if<std::is_same<T, U>::value &&
                            std::is_trivially_destructible<U>::value>::type
    destroy(U *)
    {
    }

    // forward everything else to the base
    using base_allocator::construct;
    using base_allocator::destroy;
};

template <typename T, typename base_allocator = std::allocator<T>>
using uninitialized_vector =
    std::vector<T, uninitialized_allocator<T, base_allocator>>;

#endif
